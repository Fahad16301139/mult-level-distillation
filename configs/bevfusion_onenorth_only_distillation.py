_base_ = [
    '../_base_/models/bevfusion_lidar_voxel01_second_secfpn.py',
    '../_base_/default_runtime.py'
]

# Custom dataset settings for Singapore OneNorth ONLY
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# 🏙️ SINGAPORE ONENORTH ONLY: Split 39 samples into 30 train + 9 val
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(
        type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# 🎯 ONENORTH-ONLY Data Config with Internal Split
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train_singapore_onenorth.pkl',  # 39 samples total
        pipeline=train_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        # 🔥 SPLIT: Use first 30 samples for training (77%)
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        backend_args=None,
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train_singapore_onenorth.pkl',  # Same file, but different samples
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        # 🔥 SPLIT: Use last 9 samples for validation (23%)
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        backend_args=None,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_train_singapore_onenorth.pkl',  # Use same for consistency
    metric='bbox')

test_evaluator = val_evaluator

# 🎯 OPTIMIZED for Small Dataset (39 samples)
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=40,              # More epochs for small data
    val_interval=5              # Validate every 5 epochs
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ⚡ LIGHTWEIGHT Optimizer Settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),  # Lower LR for small data
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_neck': dict(lr_mult=0.1)
        }),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=40,
        T_max=40,
        eta_min_ratio=1e-4,
        by_epoch=True)
]

# 🏙️ Singapore OneNorth-Specific Settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10,              # Save every 10 epochs
        max_keep_ckpts=3,
        save_best='NuScenes metric_pred_instances_3d_NuScenes_mAP',
        rule='greater'))

# 🎯 FOCUS: Singapore OneNorth Only
work_dir = 'work_dirs/bevfusion_singapore_onenorth_only'

# Model config (use lightweight)
model = dict(
    type='BEVFusion',
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=[0.1, 0.1, 0.2],        # Slightly larger voxels for speed
        max_voxels=(120000, 160000)
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1000, 1000],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            sizes=[[4.73, 2.08, 1.77]],  # car size
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi/4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_per_img=500)))

print("🏙️ Config: Singapore OneNorth ONLY (39 samples split 30+9)")
print("✅ Perfect for focused distillation experiments!") 