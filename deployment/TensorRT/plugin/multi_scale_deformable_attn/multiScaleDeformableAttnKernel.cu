//
// Created by Derry Lin on 2022/11/7.
//

#include <cstdio>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_helper.h"
#include "helper.h"
#include "multiScaleDeformableAttnKernel.h"
#include <cstdio>
#include <cuda/std/limits>
#include <unistd.h>

template <typename T>
__forceinline__ __device__ T hmax(const T &a, const T &b) {
  return max(a, b);
}

#if __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ __half hmax(const __half &a, const __half &b) {
  return __hmax(a, b);
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2 &a, const __half2 &b) {
  return __hmax2(a, b);
}
#else
template <>
__forceinline__ __device__ __half hmax(const __half &a, const __half &b) {
  return __hgt(a, b) ? a : b;
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2 &a, const __half2 &b) {
  return __hfma2(__hgt2(a, b), a, __hmul2(__hle2(a, b), b));
}
#endif

template <typename T> __forceinline__ __device__ T sign_05(T x) {
  if (x > 0) {
    return 0.5f;
  }
  return -0.5f;
}

template <typename T> __forceinline__ __device__ int8_t T2int8(T a) {
  a = a > 127 ? 127 : a;
  a = a < -128 ? -128 : a;
  return int8_t(a + sign_05<T>(a));
}

template <> __forceinline__ __device__ int8_t T2int8(__half a) {
  short temp = __half2short_rn(a);
  temp = temp > static_cast<short>(127) ? static_cast<short>(127) : temp;
  temp = temp < static_cast<short>(-128) ? static_cast<short>(-128) : temp;
  return static_cast<int8_t>(temp);
}

template <typename T> __forceinline__ __device__ uint8_t T2uint8(T a) {
  a = a > 255 ? 255 : a;
  a = a < 0 ? 0 : a;
  return uint8_t(a + 0.5);
}

template <> __forceinline__ __device__ uint8_t T2uint8(__half a) {
  unsigned short temp = __half2ushort_rn(a);
  temp = temp > static_cast<short>(255) ? static_cast<short>(255) : temp;
  return static_cast<uint8_t>(temp);
}

__forceinline__ __device__ int8_t half2int8(const __half &hval,
                                            const float &scale) {
  __half ret = __hdiv(hval, __float2half(scale));
  return T2int8<__half>(ret);
}

__forceinline__ __device__ uint8_t half2uint8(const __half &hval,
                                              const float &scale) {
  __half ret = __hdiv(hval, __float2half(scale));
  return T2uint8<__half>(ret);
}

__forceinline__ __device__ void qmulf(const int32_4 &a, int8_4 &c,
                                      const float &b) {
  c.x = T2int8<float>(a.x * b);
  c.y = T2int8<float>(a.y * b);
  c.z = T2int8<float>(a.z * b);
  c.w = T2int8<float>(a.w * b);
}

__forceinline__ __device__ void qmulh(const int32_4 &a, int8_4 &c,
                                      const __half &b) {
  c.x = T2int8<__half>(__int2half_rn(a.x) * b);
  c.y = T2int8<__half>(__int2half_rn(a.y) * b);
  c.z = T2int8<__half>(__int2half_rn(a.z) * b);
  c.w = T2int8<__half>(__int2half_rn(a.w) * b);
}

__forceinline__ __device__ void dp4a(const int32_t *a, const int32_t *b,
                                     int32_t &c) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(*a), "r"(*b), "r"(c));
#else
  auto ap = (int8_4 *)a, bp = (int8_4 *)b;

  c += ap->x * bp->x;
  c += ap->y * bp->y;
  c += ap->z * bp->z;
  c += ap->w * bp->w;
#endif
}

__forceinline__ __device__ void dp4a(const int32_t *a, const uint32_t *b,
                                     int32_t &c) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.u32 %0, %1, %2, %3;" : "+r"(c) : "r"(*a), "r"(*b), "r"(c));
#else
  auto ap = (int8_4 *)a;
  auto bp = (uint8_4 *)b;

  c += ap->x * bp->x;
  c += ap->y * bp->y;
  c += ap->z * bp->z;
  c += ap->w * bp->w;
#endif
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ __half ms_deform_attn_im2col_bilinear(
    const __half *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half &h, 
    const __half &w, const int &m, const int &c) {
  
  const int h_low = __half2int_rn(h);
  const int w_low = __half2int_rn(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const __half lh = __hsub(h, __int2half_rn(h_low));
  const __half lw = __hsub(w, __int2half_rn(w_low));
  const __half hh = __hsub(__float2half(1.f), lh);
  const __half hw = __hsub(__float2half(1.f), lw);

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  __half v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  __half v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  __half v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  __half v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const __half w1 = __hmul(hh, hw);
  const __half w2 = __hmul(hh, lw);
  const __half w3 = __hmul(lh, hw);
  const __half w4 = __hmul(lh, lw);

  return __hadd(__hadd(__hmul(w1, v1), __hmul(w2, v2)),
                __hadd(__hmul(w3, v3), __hmul(w4, v4)));
}




template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t *data_value, 
    const int *data_spatial_shapes,
    const int *data_level_start_index, 
    const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, 
    const int batch_size,
    const int spatial_size, 
    const int num_heads, 
    const int channels,
    const int num_levels, 
    const int num_query, 
    const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;
    
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      
      const scalar_t *data_value_ptr =
          data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w, num_heads, channels,
                                                h_im, w_im, m_col, c_col) * weight;
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, 
    const __half *data_value, 
    const int *data_spatial_shapes,
    const int *data_level_start_index, 
    const __half *data_sampling_loc,
    const __half *data_attn_weight, 
    const int batch_size, 
    const int spatial_size,
    const int num_heads, 
    const int channels, 
    const int num_levels,
    const int num_query, 
    const int num_point, 
    __half *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    __half *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    __half col = 0;
    
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      
      const __half *data_value_ptr =
          data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const __half loc_w = data_sampling_loc[data_loc_w_ptr];
        const __half loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const __half weight = data_attn_weight[data_weight_ptr];

        const __half h_im = __hsub(__hmul(loc_h, __int2half_rn(spatial_h)), __float2half(0.5f));
        const __half w_im = __hsub(__hmul(loc_w, __int2half_rn(spatial_w)), __float2half(0.5f));

        if (__hgt(h_im, __float2half(-1.f)) && __hgt(w_im, __float2half(-1.f)) &&
            __hlt(h_im, __int2half_rn(spatial_h)) && __hlt(w_im, __int2half_rn(spatial_w))) {
          // Pass data_value_ptr by reference
          col += ms_deform_attn_im2col_bilinear(data_value_ptr,
                                                   spatial_h, 
                                                   spatial_w, 
                                                   num_heads, 
                                                   channels, 
                                                   h_im, w_im, 
                                                   m_col, c_col) * weight;
        }
        
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}



template <typename scalar_t>
void ms_deformable_im2col_cuda(const scalar_t *data_value,
                               const int *data_spatial_shapes,
                               const int *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, 
                               const int spatial_size,
                               const int num_heads,
                               const int channels,
                               const int num_levels, 
                               const int num_query,
                               const int num_point,
                               scalar_t *data_col, 
                               cudaStream_t stream) {
  
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = THREADS_PER_BLOCK;
  // printf("%d", data_spatial_shapes);
  cudaMemset((scalar_t *)data_col, 0, num_kernels * sizeof(scalar_t));
  
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), num_threads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, 
          data_level_start_index, data_sampling_loc, 
          data_attn_weight, batch_size, 
          spatial_size, num_heads, channels, 
          num_levels, num_query, num_point,
          data_col);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}


template <>
void ms_deformable_im2col_cuda(
    const __half *data_value, const int *data_spatial_shapes,
    const int *data_level_start_index, const __half *data_sampling_loc,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    __half *data_col, cudaStream_t stream) {
  
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = THREADS_PER_BLOCK;
  
  // Initialize output memory
  cudaMemset(data_col, 0, num_kernels * sizeof(__half));

  ms_deformable_im2col_gpu_kernel<__half>
      <<<GET_BLOCKS(num_kernels), num_threads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, 
          data_level_start_index, data_sampling_loc, 
          data_attn_weight, batch_size, spatial_size, 
          num_heads, channels, num_levels, num_query, num_point,
          data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}


template void ms_deformable_im2col_cuda<float>(
    const float *data_value, 
    const int *data_spatial_shapes,
    const int *data_level_start_index, 
    const float *data_sampling_loc,
    const float *data_attn_weight, 
    const int batch_size, 
    const int spatial_size,
    const int num_heads, 
    const int channels, 
    const int num_levels,
    const int num_query, 
    const int num_point,
    float *data_col, 
    cudaStream_t stream);


template void ms_deformable_im2col_cuda<__half>(
    const __half *data_value, 
    const int *data_spatial_shapes,
    const int *data_level_start_index, 
    const __half *data_sampling_loc,
    const __half *data_attn_weight, 
    const int batch_size,
    const int spatial_size, 
    const int num_heads, 
    const int channels,
    const int num_levels, 
    const int num_query, 
    const int num_point,
    __half *data_col, 
    cudaStream_t stream);
