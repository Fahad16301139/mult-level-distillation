# Teacher Model Configuration for Knowledge Distillation - Full Mini nuScenes
# Uses the full BEVFusion model from checkpoints/epoch_20.pth as teacher

# Inherit base config from the BEVFusion project
_base_ = ['../projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# Work directory for teacher model
work_dir = 'work_dirs/bevfusion_teacher_mini'

# Load from the full model checkpoint in checkpoints folder
load_from = 'checkpoints/epoch_20.pth'

# ============================================================================
# FULL TEACHER MODEL CONFIGURATION - ORIGINAL BEVFUSION ARCHITECTURE
# ============================================================================

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,  # FULL: Original value (teacher should have more)
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],  # FULL: Original smaller voxels (higher resolution)
            max_voxels=[120000, 160000],  # FULL: Original higher capacity
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1408, 1408, 40],  # FULL: Original higher resolution
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 256)),  # 🔧 FIXED: 256 final layer to match checkpoint
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,  # 🔧 FIXED: Matches middle encoder output (256)
        out_channels=[128, 256],  # FULL: Original channels (vs [64, 128] in student)
        layer_nums=[5, 5],  # FULL: Original layer count (vs [3, 3] in student)
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],  # FULL: Matches backbone output
        out_channels=[256, 256],  # FULL: Original channels (vs [128, 128] in student)
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,  # FULL: Original proposal count (vs 100 in student)
        auxiliary=True,
        in_channels=512,  # FULL: Sum of neck outputs (256 + 256 = 512)
        hidden_channel=128,  # FULL: Original transformer dimension (vs 64 in student)
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),  # FULL: Original 128 dims, 8 heads (vs 64 dims, 4 heads in student)
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),  # FULL: Original 128 dims, 8 heads (vs 64 dims, 4 heads in student)
            ffn_cfg=dict(
                embed_dims=128,  # FULL: Original dimension (vs 64 in student)
                feedforward_channels=256,  # FULL: Original FFN capacity (vs 128 in student)
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),  # FULL: Original 128 (vs 64 in student)
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1408, 1408, 40],  # FULL: Matches sparse_shape
            voxel_size=[0.075, 0.075, 0.2],  # FULL: Matches preprocessor
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
            grid_size=[1408, 1408, 40],  # FULL: Matches sparse_shape
            out_size_factor=8,
            voxel_size=[0.075, 0.075],  # FULL: Matches preprocessor XY
            pc_range=[-54.0, -54.0],
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],  # FULL: Matches preprocessor XY
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
    # Set to evaluation mode - this is a teacher model that won't be trained further
    train_cfg=None)

# ============================================================================
# VALIDATION DATALOADER - FULL MINI NUSCENES
# ============================================================================

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                type='LoadPointsFromMultiSweeps',
                use_dim=5),
            dict(
                point_cloud_range=[
                    -54.0,
                    -54.0,
                    -5.0,
                    54.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'num_pts_feats',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

# ============================================================================
# VALIDATION EVALUATOR - FULL MINI NUSCENES
# ============================================================================

val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric',
)

# ============================================================================
# TEST DATALOADER - FULL MINI NUSCENES
# ============================================================================

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                type='LoadPointsFromMultiSweeps',
                use_dim=5),
            dict(
                point_cloud_range=[
                    -54.0,
                    -54.0,
                    -5.0,
                    54.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'num_pts_feats',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

# ============================================================================
# TRAINING DATALOADER - FULL MINI NUSCENES
# ============================================================================

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='NuScenesDataset',
        ann_file='nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                type='LoadPointsFromMultiSweeps',
                use_dim=5),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    -0.78539816,
                    0.78539816,
                ],
                scale_ratio_range=[
                    0.9,
                    1.1,
                ],
                translation_std=0.5,
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -54.0,
                    -54.0,
                    -5.0,
                    54.0,
                    54.0,
                    3.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -54.0,
                    -54.0,
                    -5.0,
                    54.0,
                    54.0,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'construction_vehicle',
                    'bus',
                    'trailer',
                    'barrier',
                    'motorcycle',
                    'bicycle',
                    'pedestrian',
                    'traffic_cone',
                ],
                type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                meta_keys=[
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'transformation_3d_flow',
                    'pcd_rotation',
                    'pcd_scale_factor',
                    'pcd_trans',
                    'num_pts_feats',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        use_valid_flag=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

test_evaluator = val_evaluator 