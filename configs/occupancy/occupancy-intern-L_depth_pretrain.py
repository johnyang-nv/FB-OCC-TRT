# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (960, 1760),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
use_checkpoint = True
sync_bn = True
# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [2.0, 42.0, 0.5],
}
depth_categories = 80 #(grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]



bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)


num_Z_anchors = 8
voxel_size = [0.1, 0.1, 0.1]

bev_h_ = 100
bev_w_ = 100
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_ * 2
_num_levels_= 1
numC_Trans=80


empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False
img_norm_cfg = None

cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False
occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channel = 256
voxel_channels = [64, 64*2, 64*4]

downsample=8
model = dict(
    type='NewBEV',
    use_depth_supervision=True,
    depth_pretrain=True,
    img_backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=use_checkpoint,
        out_indices=(1, 2, 3),
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[320, 640, 1280],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=downsample,
        se_depth_map=True,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        use_dcn=False,
    ),

    img_view_transformer=None,
    frpn=None,
    bevformer_encoder=None,
    img_bev_encoder_backbone=None,
    img_bev_encoder_neck=None,
    occupancy_head=None,
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        normalize_cfg=dict(
             mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=True
        )
        ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        dtype='float16',
        use_dim=[0, 1, 2, 3],
        translate2ego=True,
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy', ignore_nonvisible=True, filter_lidar_points=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    #
    dict(type='LoadBEVMask', point_cloud_range=point_cloud_range, bev_size=(bev_h_, bev_w_)),


    # dict(type='PadMultiViewImage'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_mask', 'gt_depth'
                               ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        dtype='float16',
        use_dim=[0, 1, 2, 3],
        translate2ego=True,
        file_client_args=file_client_args),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy', ignore_nonvisible=True, filter_lidar_points=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config), #
    dict(type='LoadBEVMask', point_cloud_range=point_cloud_range, bev_size=(bev_h_, bev_w_)),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'gt_bev_mask', 'gt_occupancy', 'visible_mask', 'gt_depth'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1.e-4, weight_decay=5e-2,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=37, layer_decay_rate=0.90,
                       depths=[5, 5, 22, 5], offset_lr_scale=0.01))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='WechatLoggerHook'),
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
]
load_from = '/mount/data/work_dirs/occupancy-intern-L_depth_pretrain/epoch_2.pth'
# evaluation = dict(interval=1, pipeline=test_pipeline)
# fp16 = dict(loss_scale='dynamic')
# checkpoint_config = dict(interval=5)
# find_unused_parameters=True

# Input shape: (256, 704)
# Flops: 192.3 GFLOPs
# Params: 58.39 M
# find_unused_parameters=True