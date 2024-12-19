# Copyright (c) 2022-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn.bricks.registry import ATTENTION         
from mmcv.runner import force_fp32
from mmcv.utils import ext_loader

from mmdet3d.models.fbbev.view_transformation.backward_projection.bevformer_utils.spatial_cross_attention_depth \
    import DA_SpatialCrossAttention, DA_MSDeformableAttention

from deployment.utils.trt_register import TRT_FUNCTIONS

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@ATTENTION.register_module()
class DA_SpatialCrossAttentionTRT(DA_SpatialCrossAttention):    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                level_start_index=None,
                flag='encoder',
                bev_query_depth=None,
                pred_img_depth=None,
                bev_mask=None,
                per_cam_mask_list=None,
                indexes=None, 
                queries_rebatch=None,
                reference_points_rebatch=None,
                bev_query_depth_rebatch=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        N, B, len_query, Z, _ = bev_query_depth.shape
        B, N, DC, H, W = pred_img_depth.shape
        bev_query_depth = bev_query_depth.permute(1, 0, 2, 3, 4) 
        pred_img_depth = pred_img_depth.view(B*N, DC, H, W)
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)

        

        if bev_mask is not None:
            per_cam_mask_list_ = per_cam_mask_list & bev_mask[None, :, :, None]
        else:
            per_cam_mask_list_ = per_cam_mask_list
            
        max_len = indexes.size(2) 
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[j, i, indexes[j, i].nonzero()][:, 0]
                dims = torch.arange(index_query_per_img.shape[0])
                queries_rebatch[j, i, dims, :] = query[j, index_query_per_img, :]
                bev_query_depth_rebatch[j, i, dims, :, :] = bev_query_depth[j, i, index_query_per_img, :, :]
                reference_points_rebatch[j, i, dims, :, :] = reference_points_per_img[j, index_query_per_img, :, :]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        bev_query_depth_rebatch = (bev_query_depth_rebatch - self.dbound[0]) / self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, DC-1).to(torch.long)
        bev_query_depth_rebatch = bev_query_depth_rebatch.view(bev_query_depth_rebatch.size(0),
                                                               bev_query_depth_rebatch.size(1),
                                                               bev_query_depth_rebatch.size(2),
                                                               bev_query_depth_rebatch.size(3))
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch, num_classes=DC)        
                                   
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,\
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,\
                                            level_start_index=level_start_index,\
                                            bev_query_depth=bev_query_depth_rebatch.view(bs*self.num_cams, max_len, D, DC),\
                                            pred_img_depth=pred_img_depth, \
                                            ).view(bs, self.num_cams, max_len, self.embed_dims)

        bs, num_cams, max_len, feature_dim = queries.size()
        
        for j in range(bs):
            for i, per_cam_mask in enumerate(per_cam_mask_list_):
                index_query_per_img = indexes[j, i, indexes[j, i].nonzero()][:, 0]
                dims = torch.arange(index_query_per_img.shape[0])
                slots[j, index_query_per_img, :] += queries[j, i, dims, :]

        count = per_cam_mask_list_.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        if self.layer_scale is None:
            return self.dropout(slots) + inp_residual
        else:
            return  self.dropout(self.layer_scale * slots) +  inp_residual


@ATTENTION.register_module()
class DA_MSDeformableAttentionTRT(DA_MSDeformableAttention): 
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 num_Z_anchors=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 disable_deformable=False,
                 norm_cfg=None,
                 init_cfg=None):
        super(DA_MSDeformableAttentionTRT, self).__init__(embed_dims,
                                                          num_heads,
                                                          num_levels,
                                                          num_points,
                                                          num_Z_anchors,
                                                          im2col_step,
                                                          dropout,
                                                          batch_first,
                                                          disable_deformable,
                                                          norm_cfg,
                                                          init_cfg)
        self.multi_scale_deformable_attn = TRT_FUNCTIONS.get("multi_scale_deformable_attn")
        self.init_weights()
    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,
               
                **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        if self.disable_deformable:
            sampling_offsets = sampling_offsets * 0
            attention_weights = attention_weights * 0
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
            
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]

            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        depth_reference_points = reference_points.reshape(bs, num_query * num_Z_anchors, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[...,0]).contiguous()
        
        depth_weights = self.multi_scale_deformable_attn(pred_img_depth.unsqueeze(2).contiguous(), 
                                                         spatial_shapes[0:1], 
                                                         level_start_index[0:1], 
                                                         depth_reference_points,
                                                         depth_attention_weights).reshape(bs, num_query, num_Z_anchors, -1)
        
        depth_weights = (depth_weights * bev_query_depth).sum(-1)
        depth_weights = depth_weights.unsqueeze(2).repeat(1,1, num_points, 1).reshape(bs, num_query, num_all_points)
        
        attention_weights = attention_weights * depth_weights[:, :, None, None, :]
        
        output = self.multi_scale_deformable_attn(value, 
                                                  spatial_shapes, 
                                                  level_start_index, 
                                                  sampling_locations, 
                                                  attention_weights)     

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return output
    