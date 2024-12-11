from deployment.trt_functions.grid_sampler import grid_sampler, grid_sampler2
from deployment.trt_functions.multi_scale_deformable_attn import (
    multi_scale_deformable_attn,
    multi_scale_deformable_attn2,
)
from deployment.trt_functions.modulated_deformable_conv2d import (
    modulated_deformable_conv2d,
    modulated_deformable_conv2d2,
)
from deployment.trt_functions.rotate import rotate, rotate2
from deployment.trt_functions.inverse import inverse
from deployment.trt_functions.bev_pool_v2 import bev_pool_v2, bev_pool_v2_2, bev_pool_v2_gpu
from deployment.trt_functions.multi_head_attn import qkv, qkv2
from deployment.utils.trt_register import TRT_FUNCTIONS


TRT_FUNCTIONS.register_module(module=grid_sampler)
TRT_FUNCTIONS.register_module(module=grid_sampler2)

TRT_FUNCTIONS.register_module(module=multi_scale_deformable_attn)
TRT_FUNCTIONS.register_module(module=multi_scale_deformable_attn2)

TRT_FUNCTIONS.register_module(module=modulated_deformable_conv2d)
TRT_FUNCTIONS.register_module(module=modulated_deformable_conv2d2)

TRT_FUNCTIONS.register_module(module=rotate)
TRT_FUNCTIONS.register_module(module=rotate2)

TRT_FUNCTIONS.register_module(module=inverse)

TRT_FUNCTIONS.register_module(module=bev_pool_v2)
TRT_FUNCTIONS.register_module(module=bev_pool_v2_2)

TRT_FUNCTIONS.register_module(module=qkv)
TRT_FUNCTIONS.register_module(module=qkv2)

