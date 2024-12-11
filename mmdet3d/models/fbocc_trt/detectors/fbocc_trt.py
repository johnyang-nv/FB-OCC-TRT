import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from mmdet.models import DETECTORS
from mmdet3d.models.fbbev.detectors.fbocc import FBOCC, generate_forward_transformation_matrix

from deployment.utils.trt_register import TRT_FUNCTIONS

@DETECTORS.register_module()
class FBOCCTRT(FBOCC):
    def __init__(self, *args, **kwargs):
        super(FBOCCTRT, self).__init__(*args, **kwargs)
        self.bev_pool_v2 = TRT_FUNCTIONS.get("bev_pool_v2")
        self.inverse = TRT_FUNCTIONS.get("inverse")
        self.grid_sample = TRT_FUNCTIONS.get("grid_sampler")
        self.bev_h, self.bev_w = self.backward_projection.bev_h, self.backward_projection.bev_w
        self.x_bound = self.backward_projection.transformer.encoder.x_bound
        self.y_bound = self.backward_projection.transformer.encoder.y_bound
        self.z_bound = self.backward_projection.transformer.encoder.z_bound
        self.pc_range = self.backward_projection.transformer.encoder.pc_range
        self.final_dim = self.backward_projection.transformer.encoder.final_dim
        
        self.num_cams = 6
        self.embed_dims = self.backward_projection.transformer.embed_dims
    
    def prepare_mlp_inputs(self, cam_params):
        return self.depth_net.get_mlp_input(*cam_params)
    
    def generate_forward_augs(self, bda):
        return generate_forward_transformation_matrix(bda)
    
    def prepare_grid(self, bda, history_forward_augs, forward_augs, curr_to_prev_ego_rt, start_of_sequence):
        n, mc, z, h, w = 1, 80, 8, 100, 100
        xs = torch.linspace(0, w - 1, w).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.

        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        rt_flow = (self.inverse(feat2bev) @ history_forward_augs @ curr_to_prev_ego_rt @ self.inverse(forward_augs) @ feat2bev)
        
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0])
        grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        return grid
    
    def prepare_bevpool_inputs(self, cam_params):
        coor = self.forward_projection.get_lidar_coor(*cam_params)
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
            self.forward_projection.voxel_pooling_prepare_v2(coor)
        return ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths
    
    def prepare_bwdproj_inputs(self, cam_params, bs=1):
        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, dim='2d', bs=bs)
        ref_3d = self.get_reference_points(self.bev_h, self.bev_w, self.pc_range[5]-self.pc_range[2], dim='3d', bs=bs)
        ref_3d, reference_points_cam, per_cam_mask_list, bev_query_depth = self.point_sampling_trt(ref_3d, cam_params)

        indexes, index_len, queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch \
              = self.get_rebatch_tensors_N_indices(per_cam_mask_list, reference_points_cam, bs=bs)
        
        return ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, index_len, \
            queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch
    
    def get_rebatch_tensors_N_indices(self, per_cam_mask_list, reference_points_cam, bs=1):
        indexes = [[] for _ in range(bs)]

        index_len = torch.zeros(bs, len(per_cam_mask_list))

        for j in range(bs):
            for i, per_cam_mask in enumerate(per_cam_mask_list):
                index_query_per_img = per_cam_mask[j].sum(-1).nonzero()
                index_query_per_img = index_query_per_img[:, 0]
                if len(index_query_per_img) == 0:
                    index_query_per_img = per_cam_mask_list[i][j].sum(-1).nonzero()
                    index_query_per_img = index_query_per_img.view(index_query_per_img.size(0))[0:1]
                indexes[j].append(index_query_per_img)
                index_len[j, i] = int(len(index_query_per_img))
            indexes[j] = rnn_utils.pad_sequence(indexes[j], batch_first=True)

        indexes = torch.cat(indexes, dim=0).view(bs, len(per_cam_mask_list), -1)
        index_len = index_len.view(bs, -1).to(torch.int32)
        max_len = index_len.max().to(torch.int32)

        D = reference_points_cam.size(3)
        queries_rebatch = torch.zeros(
            [bs, self.num_cams, max_len.int(), self.embed_dims])
        reference_points_rebatch = torch.zeros(
            [bs, self.num_cams, max_len.int(), D, 2])
        bev_query_depth_rebatch = torch.zeros(
            [bs, self.num_cams, max_len, D, 1])

        return indexes, index_len, queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch
    
    def get_reference_points(self, H, W, Z=8, dim='3d', bs=1, device='cpu', dtype=torch.float):
        if dim == '3d':
            X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1]/2
            Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1]/2
            Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1]/2
            Y, X, Z = torch.meshgrid([Y, X, Z])
            coords = torch.stack([X, Y, Z], dim=-1)
            coords = coords.to(dtype).to(device)
            return coords
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        
    def point_sampling_trt(self, reference_points, cam_params):
        return self.backward_projection.transformer.encoder.point_sampling(
            reference_points, pc_range=None, img_metas=None, cam_params=cam_params)
    
    def forward_trt(self, inputs):
        imgs, mlp_input, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, \
            ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, \
                queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch, \
                start_of_sequence, grid, history_bev, history_seq_ids, history_sweep_time = inputs
        
        # image encoder
        x = self.image_encoder(imgs) 
        feat, depth = self.depth_net(x, mlp_input) 
        
        # forward projection
        bev_feat = self.bev_pool_v2(depth, 
                                    feat.permute(0, 1, 3, 4, 2), 
                                    ranks_depth, 
                                    ranks_feat, 
                                    ranks_bev,  
                                    interval_starts, 
                                    interval_lengths).permute(0, 4, 2, 3, 1)
        
        # backward_projection
        bev_feat_refined = self.backward_projection_trt(feat,
                                    [ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, \
                                     queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch],
                                    lss_bev=bev_feat.mean(-1),
                                    bev_mask=None,
                                    gt_bboxes_3d=None, 
                                    pred_img_depth=depth)  
        bev_feat = bev_feat_refined[..., None] + bev_feat 
        
        # Fuse History
        bev_feat, output_history_bev, output_history_seq_ids, output_history_sweep_time = self.fuse_history_trt(bev_feat, 
                                         start_of_sequence, 
                                         history_bev, 
                                         history_sweep_time, 
                                         history_seq_ids,
                                         grid) 
        
        bev_feat = self.bev_encoder(bev_feat) 
        pred_occupancy = self.occupancy_head(bev_feat, results={})['output_voxels'][0] 
        
        return pred_occupancy, output_history_bev, output_history_seq_ids, output_history_sweep_time
    
    def fuse_history_trt(self, 
                         curr_bev, 
                         start_of_sequence, 
                         history_bev, 
                         history_sweep_time, 
                         history_seq_ids,
                         grid
                         ):
        curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w

        ## Consider First batch as a start_of_sequence
        ## Replace all the new sequences' positions in history with the curr_bev information
        history_bev = history_bev.to(curr_bev)
        history_bev = start_of_sequence.float() * curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) + (1. - start_of_sequence.float()) * history_bev
        
        n, c_, z, h, w = curr_bev.shape # 1 80 8 100 100
        
        tmp_bev = history_bev
        n, mc, z, h, w = tmp_bev.shape
        tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        
        sampled_history_bev = self.grid_sample(tmp_bev, 10. * grid.to(curr_bev.dtype).permute(0, 4, 3, 1, 2), 
                                               align_corners=True, interpolation_mode=self.interpolation_mode, padding_mode='zeros')
        
        ## Update history
        # Add in current frame to features & timestep
        history_sweep_time = torch.cat([history_sweep_time.new_zeros(history_sweep_time.shape[0], 1), history_sweep_time], dim=1) # B x (1 + T)
            
        sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
        curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 
        
        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        
        feats_to_return = torch.cat(
        [feats_to_return, history_sweep_time[:, :, None, None, None, None].repeat(
            1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
        ], dim=2) # B x (1 + T) x 81 x Z x H x W

        # Time conv
        B, Tplus1, C, Z, H, W = feats_to_return.shape
        feats_to_return = self.history_keyframe_time_conv(feats_to_return.reshape(B*Tplus1, C, Z, H, W))
        feats_to_return = feats_to_return.reshape(B, Tplus1, 80, Z, H, W) # B x (1 + T) x 80 xZ x H x W
        
        # Cat keyframes & conv
        B, Tplus1, C, Z, H, W = feats_to_return.shape # 1 x 17 x 80 x 8 x 100 x 100
        feats_to_return = feats_to_return.reshape(B, Tplus1*C, Z, H, W)
        feats_to_return = self.history_keyframe_cat_conv(feats_to_return) # B x C x H x W or B x C x Z x H x W
        
        history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        history_sweep_time = history_sweep_time[:, :-1]

        feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)

        return feats_to_return.clone(), history_bev, history_seq_ids, history_sweep_time

    def backward_projection_trt(self, feat, bev_info, lss_bev=None, gt_bboxes_3d=None, pred_img_depth=None, bev_mask=None):
        bs, num_cam, c, h, w = feat.shape
        dtype = feat.dtype
        bev_queries = self.backward_projection.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1)
        
        lss_bev = lss_bev.flatten(2).permute(2, 0, 1)
        bev_queries = bev_queries + lss_bev
        
        bev_pos = self.backward_projection.positional_encoding(bs, self.backward_projection.bev_h, self.backward_projection.bev_w, bev_queries.device).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        spatial_shapes = [(h, w)]
        feat_flatten = feat.flatten(3).permute(1, 0, 3, 2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  
        
        bev_embed = self.back_proj_transformer_encoder_trt(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_info,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            gt_bboxes_3d=gt_bboxes_3d,
            pred_img_depth=pred_img_depth,
            prev_bev=None,
            bev_mask=bev_mask
        )
        bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w).contiguous()
        return bev_embed

    def back_proj_transformer_encoder_trt(self, 
                                bev_query,
                                key,
                                value,
                                bev_info,
                                *args,
                                bev_h=None,
                                bev_w=None,
                                bev_pos=None,
                                spatial_shapes=None,
                                level_start_index=None,
                                valid_ratios=None,
                                gt_bboxes_3d=None,
                                pred_img_depth=None,
                                bev_mask=None,
                                prev_bev=None,
                                **kwargs):
                        
        output = bev_query
        intermediate = []
        ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, \
            queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch = bev_info
        bev_query = bev_query.permute(1, 0, 2) # [10000, bs, 80]
        bev_pos = bev_pos.permute(1, 0, 2) # [bs, 10000, 80]        
                
        for lid, layer in enumerate(self.backward_projection.transformer.encoder.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=None,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                prev_bev=prev_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                per_cam_mask_list=per_cam_mask_list,
                bev_mask=bev_mask,
                bev_query_depth=bev_query_depth,
                pred_img_depth=pred_img_depth,
                indexes=indexes,  
                queries_rebatch=queries_rebatch,
                reference_points_rebatch=reference_points_rebatch,
                bev_query_depth_rebatch=bev_query_depth_rebatch,
                **kwargs)

            bev_query = output
            if self.backward_projection.transformer.encoder.return_intermediate:
                intermediate.append(output)

        if self.backward_projection.transformer.encoder.return_intermediate:
            return torch.stack(intermediate)
        return output
    
    def post_process(self, pred_occupancy):
        pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
        pred_occupancy = pred_occupancy[..., 1:]     
        pred_occupancy = pred_occupancy.softmax(-1)

        # convert to CVPR2023 Format
        pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
        pred_occupancy = torch.flip(pred_occupancy, [2])
        pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
        pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
        
        pred_occupancy_category = pred_occupancy.argmax(-1) 
        
        pred_occupancy_category= pred_occupancy_category.cpu().numpy()
        
        result_dict = {}
        result_dict['pts_bbox'] = None
        result_dict['iou'] = None
        result_dict['pred_occupancy'] = pred_occupancy_category
        result_dict['index'] = None
        return [result_dict]
    
