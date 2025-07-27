_base_ = [
    'mmdet3d/configs/_base_/datasets/nus-3d.py',
    'mmdet3d/configs/_base_/schedules/cyclic_20e.py', 
    'mmdet3d/configs/_base_/default_runtime.py',
]

# 🔥 CRITICAL FIX: Use FULL NuScenes dataset (not mini!) for proper training
# 🔥 CRITICAL FIX: Use SAME domain as teacher (Boston) to eliminate domain gap

auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

custom_imports = dict(
    allow_failed_imports=False, 
    imports=['projects.BEVFusion.bevfusion']
)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP'
)
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'

# ============================================================================
# LIGHTWEIGHT STUDENT MODEL (256 channel neck)
# ============================================================================

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,  # Reduced from 10 for efficiency
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.1, 0.1, 0.2],  # Slightly larger than teacher (0.075)
            max_voxels=[80000, 90000],  # Reduced from teacher's [120000, 160000]
            voxelize_reduce=True
        )
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1088, 1088, 40],  
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 128)),  # Last layer 128 (reduced from 256)
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[64, 128],  # Reduced from [128, 256]
        layer_nums=[3, 3],  # Reduced from [5, 5]
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],  # Matches backbone output
        out_channels=[128, 128],  # 🎯 CRITICAL: Student neck outputs 128+128=256 total channels
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,  # Sum of neck channels: 128+128=256
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-54.0, -54.0, -1.80032795, 54.0, 54.0, -1.80032795],
                [-54.0, -54.0, -1.74440365, 54.0, 54.0, -1.74440365],
                [-54.0, -54.0, -1.68526504, 54.0, 54.0, -1.68526504],
                [-54.0, -54.0, -1.67339111, 54.0, 54.0, -1.67339111],
                [-54.0, -54.0, -1.61785072, 54.0, 54.0, -1.61785072],
                [-54.0, -54.0, -1.80984986, 54.0, 54.0, -1.80984986],
                [-54.0, -54.0, -1.76396500, 54.0, 54.0, -1.76396500],
                [-54.0, -54.0, -1.73708846, 54.0, 54.0, -1.73708846],
                [-54.0, -54.0, -0.93352168, 54.0, 54.0, -0.93352168],
                [-54.0, -54.0, -0.86092367, 54.0, 54.0, -0.86092367]
            ],
            sizes=[
                [4.60718145, 1.95017717, 1.72270761],
                [6.73778078, 2.4560939, 2.73004906],
                [12.01320693, 2.87427237, 3.81509561],
                [1.68452161, 0.60058911, 1.27192197],
                [0.7256437, 0.66344886, 1.75748069],
                [0.40359262, 0.39694519, 1.06232151],
                [2.11200554, 0.77914566, 1.24902108],
                [1.75073828, 0.82877739, 1.88853754],
                [1.73698127, 0.84393492, 0.84162378],
                [2.15343563, 0.84027184, 0.90159926]
            ],
            rotations=[0, 1.57],
            reshape_out=False
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', 
            beta=0.1111111111111111, 
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.2
        )
    ),
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                )
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=4096,
            max_num=500
        )
    )
)

# ============================================================================
# DATA PIPELINES
# ============================================================================

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=5,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=0.5
    ),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    ),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    ),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=5,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats']
    )
]

# ============================================================================
# 🔥 CRITICAL FIX: USE FULL NUSCENES DATASET (NOT MINI!)
# ============================================================================

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_train.pkl',  # 🔥 FULL dataset (28,000 samples) not mini!
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_camera=False, use_lidar=True),
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        use_valid_flag=True
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',  # 🔥 FULL validation set
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_camera=False, use_lidar=True),
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',  # 🔥 FULL test set
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_camera=False, use_lidar=True),
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)

# ============================================================================
# EVALUATORS
# ============================================================================

val_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',  # 🔥 FULL validation
    metric='bbox',
    backend_args=backend_args
)

test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',  # 🔥 FULL test
    metric='bbox',
    backend_args=backend_args
)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'img_neck': dict(lr_mult=0.1, decay_mult=1.0)
        }
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=1e-7,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# 🔥 CRITICAL NOTES:
# 1. This config uses FULL NuScenes dataset (~28,000 training samples) instead of mini (~300)
# 2. Same domain as your teacher model (no Boston→Singapore gap)  
# 3. Student model has 256-channel neck output to match your distillation setup
# 4. Proper lightweight architecture with reasonable capacity 