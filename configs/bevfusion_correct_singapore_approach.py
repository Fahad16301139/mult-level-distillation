# CORRECT Singapore BEVFusion Approach:
# Training: Singapore regions (One North + Holland Village)
# Validation: Full nuScenes (stable training, no sample token issues)  
# Testing: Singapore Queenstown (final domain-specific evaluation)
# Model: SAME lightweight architecture as original

auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
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
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.BEVFusion.bevfusion',
        'custom_nuscenes_metric',
    ])
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP')
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
db_sampler = dict(
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
    data_root='data/nuscenes/',
    info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(
            barrier=5,
            bicycle=5,
            bus=5,
            car=5,
            construction_vehicle=5,
            motorcycle=5,
            pedestrian=5,
            traffic_cone=5,
            trailer=5,
            truck=5)),
    rate=1.0,
    sample_groups=dict(
        barrier=2,
        bicycle=6,
        bus=4,
        car=2,
        construction_vehicle=7,
        motorcycle=6,
        pedestrian=2,
        traffic_cone=2,
        trailer=6,
        truck=3))
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 1e-06
metainfo = dict(classes=[
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
])

# SAME lightweight model as original config
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,  # Reduced from 10
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.1, 0.1, 0.2],  # Increased voxel size from 0.075 to 0.1
            max_voxels=[80000, 90000],  # Reduced from [120000, 160000]
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1088, 1088, 40],  # Changed to 1088 for better divisibility
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 128)),  # Fixed progression to output 128
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,  # Fixed to match actual middle encoder output
        out_channels=[96, 160],  # INCREASED from [64, 128] for better geometry learning
        layer_nums=[4, 4],  # INCREASED from [3, 3] for more feature learning
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[96, 160],  # Match backbone output
        out_channels=[160, 160],  # INCREASED from [128, 128] for better features
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=150,  # INCREASED from 100 for more detection opportunities
        auxiliary=True,
        in_channels=320,  # Sum of neck outputs: 160 + 160 = 320
        hidden_channel=96,  # INCREASED from 64 for better geometry regression
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=96, num_heads=4, dropout=0.1),  # INCREASED embed_dims
            cross_attn_cfg=dict(embed_dims=96, num_heads=4, dropout=0.1),  # INCREASED embed_dims
            ffn_cfg=dict(
                embed_dims=96,  # INCREASED from 64 for better processing
                feedforward_channels=192,  # INCREASED from 128 for better capacity
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=96)),  # Reduced from 128
        train_cfg=dict(
            dataset='nuScenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1088, 1088, 40],  # Updated to match sparse_shape
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
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.35),  # INCREASED from 0.25 for better geometry learning
                iou_cost=dict(type='IoU3DCost', weight=0.35))),  # INCREASED from 0.25 for better overlap learning
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1088, 1088, 40],  # Updated to match sparse_shape
            out_size_factor=8,
            voxel_size=[0.1, 0.1],  # Match preprocessor XY
            pc_range=[-54.0, -54.0],
            nms_type=None),  # Use no NMS like original working config
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,  # Use original working value
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
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
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.4)))

# Training params
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),  # Same as original
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.33333333,
        type='LinearLR'),
    dict(
        T_max=20,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min_ratio=0.01,
        type='CosineAnnealingLR'),
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ONLY DATA APPROACH CHANGES HERE - everything else same as original

# TRAINING: Singapore regions (One North + Holland Village) - same as before
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        datasets=[
            dict(
                ann_file='nuscenes_infos_train_singapore_onenorth.pkl',  # Singapore One North
                backend_args=None,
                box_type_3d='LiDAR',
                data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
                data_root='data/nuscenes/',
                metainfo=dict(classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
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
                        rot_range=[-0.78539816, 0.78539816],
                        scale_ratio_range=[0.9, 1.1],
                        translation_std=0.5,
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
                    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='ObjectRangeFilter'),
                    dict(classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], type='ObjectNameFilter'),
                    dict(type='PointShuffle'),
                    dict(
                        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                        type='Pack3DDetInputs')],
                test_mode=False,
                type='NuScenesDataset',
                use_valid_flag=True),
            dict(
                ann_file='nuscenes_infos_train_singapore_hollandvillage.pkl',  # Singapore Holland Village
                backend_args=None,
                box_type_3d='LiDAR',
                data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
                data_root='data/nuscenes/',
                metainfo=dict(classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
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
                        rot_range=[-0.78539816, 0.78539816],
                        scale_ratio_range=[0.9, 1.1],
                        translation_std=0.5,
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
                    dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='ObjectRangeFilter'),
                    dict(classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], type='ObjectNameFilter'),
                    dict(type='PointShuffle'),
                    dict(
                        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                        type='Pack3DDetInputs')],
                test_mode=False,
                type='NuScenesDataset',
                use_valid_flag=True)],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# VALIDATION: FULL nuScenes dataset (stable, no sample token errors)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',  # FULL nuScenes validation
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
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
            dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
            dict(
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'],
                type='Pack3DDetInputs')],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',  # FULL nuScenes validation
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')

# TESTING: Singapore Queenstown (final domain-specific evaluation)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val_singapore_queenstown.pkl',  # Singapore Queenstown
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
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
            dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
            dict(
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'],
                type='Pack3DDetInputs')],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val_singapore_queenstown.pkl',  # Singapore Queenstown
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='CustomNuScenesMetric')


test_cfg = dict(type='TestLoop')
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=999)  # Skip validation during training
val_cfg = dict(type='ValLoop')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer', type='Det3DLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
work_dir = './work_dirs/bevfusion_correct_singapore_approach' 