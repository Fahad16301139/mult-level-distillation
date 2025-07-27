_base_ = [
    'mmdet3d/configs/_base_/datasets/nus-3d.py',
    'mmdet3d/configs/_base_/schedules/cyclic_20e.py',
    'mmdet3d/configs/_base_/default_runtime.py',
]

# Balanced lightweight model for Singapore data - optimized for ACCURACY while maintaining efficiency
# This config prioritizes reducing detection errors over extreme speed

backend_args = None
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Training data: Singapore regions only
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(use_camera=False, use_lidar=True)

# Balanced lightweight model - CONSERVATIVE reductions for better accuracy
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=7,  # MODERATE reduction: from 10 to 7 (was 3 in ultra-light)
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.1, 0.1, 0.2],  # MODERATE: from [0.075] to [0.1] (was 0.15 in ultra-light)
            max_voxels=[100000, 120000],  # MODERATE reduction: from [120k,160k] to [100k,120k]
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1088, 1088, 40],  # GOOD balance: maintains resolution
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 256)),  # BETTER capacity: output 256
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,  # Match middle encoder output
        out_channels=[96, 192],  # BALANCED: from [128,256] to [96,192] (better than ultra-light [32,64])
        layer_nums=[4, 4],  # MODERATE reduction: from [5,5] to [4,4] (better than ultra-light [2,2])
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[96, 192],  # Match backbone output
        out_channels=[192, 192],  # GOOD capacity: from [256,256] to [192,192]
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=150,  # BALANCED: from 200 to 150 (better than ultra-light 50)
        auxiliary=True,
        in_channels=384,  # Sum of neck outputs: 192 + 192 = 384
        hidden_channel=96,  # BALANCED: from 128 to 96 (better than ultra-light 32)
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,  # Keep minimal but effective
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=96, num_heads=6, dropout=0.1),  # GOOD capacity
            cross_attn_cfg=dict(embed_dims=96, num_heads=6, dropout=0.1),  # GOOD capacity
            ffn_cfg=dict(
                embed_dims=96,  # BALANCED: from 128 to 96
                feedforward_channels=192,  # BALANCED: from 256 to 192
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=96)),  # BALANCED
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1088, 1088, 40],  # Match sparse_shape
            voxel_size=[0.1, 0.1, 0.2],  # Match preprocessor
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1088, 1088, 40],  # Match sparse_shape
            out_size_factor=8,
            voxel_size=[0.1, 0.1],  # Match preprocessor XY
            pc_range=[-54.0, -54.0],
            nms_type='nms_gpu',  # ADD proper NMS for better detection
            nms_thr=0.2,         # NMS threshold for overlap
            score_thr=0.05,      # Score threshold for detection
            pre_maxsize=1000,    # Max detections before NMS
            post_maxsize=83),    # Max detections after NMS
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.05,  # INCREASED for better quality detections
            out_size_factor=8,
            voxel_size=[0.1, 0.1],  # Match preprocessor XY
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

# Training pipeline for Singapore data
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=5,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=0.5),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'transformation_3d_flow', 
                  'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'num_pts_feats'])
]

# Test pipeline - same as validation
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=5,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'])
]

# Training dataloader - Singapore regions
train_dataloader = dict(
    batch_size=4,  # BALANCED: not too aggressive, allows good gradient updates
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='NuScenesDataset',
                data_root='data/nuscenes/',
                ann_file='nuscenes_infos_train_singapore_onenorth.pkl',  # Singapore One North
                pipeline=train_pipeline,
                metainfo=dict(classes=class_names),
                modality=input_modality,
                test_mode=False,
                data_prefix=dict(
                    pts='samples/LIDAR_TOP',
                    sweeps='sweeps/LIDAR_TOP'),
                box_type_3d='LiDAR',
                backend_args=backend_args,
                use_valid_flag=True),
            dict(
                type='NuScenesDataset',
                data_root='data/nuscenes/',
                ann_file='nuscenes_infos_train_singapore_hollandvillage.pkl',  # Singapore Holland Village
                pipeline=train_pipeline,
                metainfo=dict(classes=class_names),
                modality=input_modality,
                test_mode=False,
                data_prefix=dict(
                    pts='samples/LIDAR_TOP',
                    sweeps='sweeps/LIDAR_TOP'),
                box_type_3d='LiDAR',
                backend_args=backend_args,
                use_valid_flag=True)
        ]))

# SOLUTION: Use FULL nuScenes validation to completely avoid sample token mismatch
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',  # FULL validation set - no sample token issues
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        test_mode=True,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR',
        backend_args=backend_args))

# Test dataloader - same as validation
test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',  # FULL validation - guaranteed compatibility
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# BETTER learning schedule for accuracy
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),  # BALANCED LR for good convergence
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate scheduler - optimized for accuracy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min_ratio=0.0001,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.8947368421052632,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default runtime settings
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Metadata
metainfo = dict(classes=class_names)
launcher = 'none'
work_dir = 'work_dirs/bevfusion_balanced_lightweight_singapore' 