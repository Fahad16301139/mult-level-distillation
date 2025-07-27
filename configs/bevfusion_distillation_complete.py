# Complete BEVFusion Distillation Configuration
# This includes ALL parameters explicitly (no _base_ inheritance)

# ============================================================================
# BASIC CONFIGURATION
# ============================================================================
auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Custom imports for BEVFusion and distillation
custom_imports = dict(
    allow_failed_imports=False, 
    imports=['projects.BEVFusion.bevfusion', 'bevfusion_distillation']
)

data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
default_scope = 'mmdet3d'
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 1e-06

metainfo = dict(classes=[
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
])

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ============================================================================
# DISTILLATION CONFIGURATION
# ============================================================================
work_dir = 'work_dirs/bevfusion_distillation_complete'
load_from = 'checkpoints/epoch_20.pth'  # Teacher checkpoint

# ============================================================================
# LIGHTWEIGHT STUDENT MODEL CONFIGURATION
# ============================================================================
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,  # REDUCED: from 10 to 5
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.1, 0.1, 0.2],  # INCREASED: from [0.075, 0.075, 0.2] to [0.1, 0.1, 0.2]
            max_voxels=[80000, 90000],  # REDUCED: from [120000, 160000]
            voxelize_reduce=True
        )
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1088, 1088, 40],  # CHANGED: from [1408, 1408, 40] to [1088, 1088, 40]
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 128)),  # REDUCED
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[64, 128],  # REDUCED: from [128, 256] to [64, 128]
        layer_nums=[3, 3],  # REDUCED: from [5, 5] to [3, 3]
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],  # MATCH: backbone output
        out_channels=[128, 128],  # REDUCED: from [256, 256] to [128, 128]
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=100,  # REDUCED: from 200 to 100
        auxiliary=True,
        in_channels=256,  # MATCH: sum of neck outputs (128 + 128 = 256)
        hidden_channel=64,  # REDUCED: from 128 to 64
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=64, num_heads=4, dropout=0.1),  # REDUCED
            cross_attn_cfg=dict(embed_dims=64, num_heads=4, dropout=0.1),  # REDUCED
            ffn_cfg=dict(
                embed_dims=64,  # REDUCED: from 128 to 64
                feedforward_channels=128,  # REDUCED: from 256 to 128
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=64)  # REDUCED
        ),
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
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
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            )
        ),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1088, 1088, 40],
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pc_range=[-54.0, -54.0],
            nms_type=None
        ),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]
        ),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=10
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', 
            reduction='mean', 
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.L1Loss', 
            reduction='mean', 
            loss_weight=0.25
        )
    )
)

# ============================================================================
# OPTIMIZER AND SCHEDULER
# ============================================================================
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),  # REDUCED learning rate
    type='OptimWrapper'
)

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.33333333, type='LinearLR'
    ),
    dict(
        T_max=20, begin=0, by_epoch=True, convert_to_iter_based=True,
        end=20, eta_min_ratio=0.0001, type='CosineAnnealingLR'
    ),
    dict(
        begin=0, by_epoch=True, convert_to_iter_based=True,
        end=8, eta_min=0.8947368421052632, type='CosineAnnealingMomentum'
    ),
    dict(
        begin=8, by_epoch=True, convert_to_iter_based=True,
        end=20, eta_min=1, type='CosineAnnealingMomentum'
    ),
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
train_cfg = dict(max_epochs=5, type='EpochBasedTrainLoop', val_interval=2)  # REDUCED for testing

# ============================================================================
# DATA PIPELINE
# ============================================================================
train_pipeline = [
    dict(backend_args=None, coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(backend_args=None, load_dim=5, pad_empty_sweeps=True, remove_close=True, sweeps_num=9, type='LoadPointsFromMultiSweeps', use_dim=5),
    dict(type='LoadAnnotations3D', with_attr_label=False, with_bbox_3d=True, with_label_3d=True),
    dict(rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.9, 1.1], translation_std=0.5, type='GlobalRotScaleTrans'),
    dict(type='RandomFlip3D'),
    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='ObjectRangeFilter'),
    dict(classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'], meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'num_pts_feats'], type='Pack3DDetInputs'),
]

test_pipeline = [
    dict(backend_args=None, coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(backend_args=None, load_dim=5, pad_empty_sweeps=True, remove_close=True, sweeps_num=9, type='LoadPointsFromMultiSweeps', use_dim=5),
    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
    dict(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'], meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'], type='Pack3DDetInputs'),
]

# ============================================================================
# DATALOADERS
# ============================================================================
train_dataloader = dict(
    batch_size=2,  # REDUCED for memory
    dataset=dict(
        type='NuScenesDataset',
        ann_file='nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True
    ),
    num_workers=2,  # REDUCED
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=test_pipeline,
        test_mode=True,
        type='NuScenesDataset'
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

test_dataloader = val_dataloader

# ============================================================================
# EVALUATION
# ============================================================================
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric',
)

test_evaluator = val_evaluator

# ============================================================================
# HOOKS
# ============================================================================
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=1,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater',
        max_keep_ckpts=3
    ),
]

# ============================================================================
# ENVIRONMENT
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)

resume = False
test_cfg = dict()
val_cfg = dict() 