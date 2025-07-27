# BEVFusion Knowledge Distillation Configuration - Standalone
# No inheritance to avoid parameter conflicts

# Basic settings
default_scope = 'mmdet3d'
custom_imports = dict(imports=['projects.BEVFusion.bevfusion', 'bevfusion_distillation_model'], allow_failed_imports=False)

# Dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# Model configuration
model = dict(
    type='BEVFusionDistillationModel',
    student_model=dict(
        type='BEVFusion',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            pad_size_divisor=32,
            voxelize_cfg=dict(
                max_num_points=5,
                point_cloud_range=point_cloud_range,
                voxel_size=[0.1, 0.1, 0.2],
                max_voxels=[80000, 90000],
                voxelize_reduce=True)),
        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
        pts_middle_encoder=dict(
            type='BEVFusionSparseEncoder',
            in_channels=5,
            sparse_shape=[1088, 1088, 40],
            order=('conv', 'norm', 'act'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 128)),
            encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
            block_type='basicblock'),
        pts_backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[64, 128],
            layer_nums=[3, 3],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),
        pts_neck=dict(
            type='SECONDFPN',
            in_channels=[64, 128],
            out_channels=[128, 128],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),
        bbox_head=dict(
            type='TransFusionHead',
            num_proposals=100,
            auxiliary=True,
            in_channels=256,
            hidden_channel=64,
            num_classes=10,
            nms_kernel_size=3,
            bn_momentum=0.1,
            num_decoder_layers=1,
            decoder_layer=dict(
                type='TransformerDecoderLayer',
                self_attn_cfg=dict(embed_dims=64, num_heads=4, dropout=0.1),
                cross_attn_cfg=dict(embed_dims=64, num_heads=4, dropout=0.1),
                ffn_cfg=dict(
                    embed_dims=64,
                    feedforward_channels=128,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)),
                norm_cfg=dict(type='LN'),
                pos_encoding_cfg=dict(input_channel=2, num_pos_feats=64)),
            train_cfg=dict(
                dataset='nuScenes',
                point_cloud_range=point_cloud_range,
                grid_size=[1088, 1088, 40],
                voxel_size=[0.1, 0.1, 0.2],
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                assigner=dict(
                    type='HungarianAssigner3D',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    cls_cost=dict(type='mmdet.FocalLossCost', gamma=2.0, alpha=0.25, weight=0.15),
                    iou_cost=dict(type='IoU3DCost', weight=0.25),
                    reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25))),
            test_cfg=dict(
                dataset='nuScenes',
                grid_size=[1088, 1088, 40],
                out_size_factor=8,
                pc_range=[-54.0, -54.0],
                voxel_size=[0.1, 0.1],
                nms_type=None),
            common_heads=dict(
                center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=[-54.0, -54.0],
                voxel_size=[0.1, 0.1],
                out_size_factor=8,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                code_size=10),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                reduction='mean'),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25, reduction='mean'),
            loss_heatmap=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0, reduction='mean'))),
    teacher_config='configs/bevfusion_teacher_mini.py',
    teacher_checkpoint='checkpoints/epoch_20.pth'
)

# Training pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=5, load_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.9, 1.1], translation_std=0.5),
    dict(type='RandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Test pipeline
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, use_dim=5, load_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Dataset configs
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        use_valid_flag=True,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# Optimization
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=20, eta_min_ratio=0.0001, begin=0, end=20, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=8, eta_min=0.8947368421052632, begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=12, eta_min=1, begin=8, end=20, by_epoch=True, convert_to_iter_based=True)
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
work_dir = 'work_dirs/bevfusion_distillation_standalone' 