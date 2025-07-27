# BEVFusion Knowledge Distillation Configuration - Full Mini nuScenes
_base_ = ['bevfusion_student_full_mini.py']

# Define the student model configuration explicitly
_student_model_config = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
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
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=64)),
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
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                iou_cost=dict(type='IoU3DCost', weight=0.25),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25))),
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1088, 1088, 40],
            out_size_factor=8,
            pc_range=[-54.0, -54.0],
            voxel_size=[0.1, 0.1],
            nms_type=None),
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
        loss_bbox=dict(
            type='mmdet.L1Loss', loss_weight=0.25, reduction='mean'),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            loss_weight=1.0,
            reduction='mean')))

# Override the model to use distillation
model = dict(
    type='BEVFusionDistillationModel',
    student_model=_student_model_config,
    teacher_config='configs/bevfusion_teacher_mini.py',
    teacher_checkpoint='checkpoints/epoch_20.pth'
)

# Update work directory
work_dir = 'work_dirs/bevfusion_distillation_full_mini'
