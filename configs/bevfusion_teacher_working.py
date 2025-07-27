# Working Teacher Model Configuration - Larger than student but CUDA-safe
# Based on the working student config but with increased capacity

_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']

# Work directory for teacher model
work_dir = 'work_dirs/bevfusion_teacher_working'

# ============================================================================
# TEACHER MODEL - LARGER THAN STUDENT BUT CUDA-SAFE
# ============================================================================

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=8,  # Teacher: 8 vs Student: 5
            point_cloud_range=[-25.6, -25.6, -3.0, 25.6, 25.6, 1.0],  # Same as student (CUDA-safe)
            voxel_size=[0.08, 0.08, 0.2],  # Teacher: 0.08 vs Student: 0.1 (slightly more voxels)
            max_voxels=[30000, 40000],  # Teacher: higher than student but not crazy
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[640, 640, 20],  # Teacher: larger than student but manageable
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 192)),  # Teacher: 192 final vs Student: 128
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=192,  # Teacher: matches middle encoder
        out_channels=[96, 192],  # Teacher: [96, 192] vs Student: [64, 128]
        layer_nums=[4, 4],  # Teacher: 4 layers vs Student: 3 layers
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[96, 192],  # Teacher: matches backbone
        out_channels=[192, 192],  # Teacher: [192, 192] vs Student: [128, 128]
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=150,  # Teacher: 150 vs Student: 100
        auxiliary=True,
        in_channels=384,  # Teacher: 192 + 192 = 384 vs Student: 256
        hidden_channel=96,  # Teacher: 96 vs Student: 64
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=96, num_heads=6, dropout=0.1),  # Teacher: 96 dims, 6 heads vs Student: 64 dims, 4 heads
            cross_attn_cfg=dict(embed_dims=96, num_heads=6, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=96,
                feedforward_channels=192,  # Teacher: 192 vs Student: 128
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=96)),
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-25.6, -25.6, -3.0, 25.6, 25.6, 1.0],  # Same as student
            grid_size=[640, 640, 20],  # Teacher: matches sparse_shape
            voxel_size=[0.08, 0.08, 0.2],  # Teacher: matches preprocessor
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
            grid_size=[640, 640, 20],
            out_size_factor=8,
            voxel_size=[0.08, 0.08],
            pc_range=[-25.6, -25.6],
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-25.6, -25.6],
            post_center_range=[-30.0, -30.0, -10.0, 30.0, 30.0, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.08, 0.08],
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
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-25.6, -25.6, -3.0, 25.6, 25.6, 1.0],
            grid_size=[640, 640, 20],
            voxel_size=[0.08, 0.08, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-25.6, -25.6],
            post_center_limit_range=[-30.0, -30.0, -10.0, 30.0, 30.0, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.08, 0.08],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=20,
        T_max=20,
        eta_min_ratio=1e-4,
        by_epoch=True)
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
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