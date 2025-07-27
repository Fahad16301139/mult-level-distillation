auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_camera=False, use_lidar=True)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,  # KEPT: efficiency gain (vs teacher 10)
            point_cloud_range=point_cloud_range,
            voxel_size=[0.1, 0.1, 0.2],  # KEPT: efficiency gain (vs teacher 0.075)
            max_voxels=[80000, 90000],  # KEPT: efficiency gain
            voxelize_reduce=True)),
    
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1088, 1088, 40],  # KEPT: efficiency gain
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 256)),  # UPGRADED: final 128→256
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock'),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],  # UPGRADED: from [64, 128] to match teacher
        layer_nums=[4, 4],  # ENHANCED: from [3, 3]
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],  # UPGRADED: Match teacher 512 total! (was [128, 128])
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=150,  # ENHANCED: from 100
        auxiliary=True,
        in_channels=512,  # UPGRADED: 256+256=512 - MATCHES TEACHER!
        hidden_channel=128,  # ENHANCED: from 64
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=6, dropout=0.1),  # ENHANCED
            cross_attn_cfg=dict(embed_dims=128, num_heads=6, dropout=0.1),  # ENHANCED
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        
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
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        
        test_cfg=dict(
            dataset='nuScenes',
            grid_size=[1088, 1088, 40],
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pc_range=[-54.0, -54.0],
            nms_type=None),
        
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=10),
        
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

# Enhanced optimizer for higher capacity
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),  # ENHANCED: 0.0001→0.0002
    type='OptimWrapper')

param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.33333333, type='LinearLR'),
    dict(T_max=20, begin=0, by_epoch=True, convert_to_iter_based=True, end=20, eta_min_ratio=0.0001, type='CosineAnnealingLR'),
    dict(begin=0, by_epoch=True, convert_to_iter_based=True, end=8, eta_min=0.8947368421052632, type='CosineAnnealingMomentum'),
    dict(begin=8, by_epoch=True, convert_to_iter_based=True, end=20, eta_min=1, type='CosineAnnealingMomentum'),
]

# Training dataset - FULL NUSCENES
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, backend_args=backend_args),
    dict(type='ObjectSample', db_sampler=dict(
        data_root=data_root, info_path=data_root + 'nuscenes_dbinfos_train.pkl', rate=1.0,
        prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(
            car=5, truck=5, bus=5, trailer=5, construction_vehicle=5,
            pedestrian=5, motorcycle=5, bicycle=5, traffic_cone=5, barrier=5)),
        sample_groups=dict(car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
            barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2),
        classes=class_names, backend_args=backend_args)),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline, metainfo=metainfo, modality=input_modality, test_mode=False,
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'), backend_args=backend_args))

val_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline, metainfo=metainfo, modality=input_modality, test_mode=True,
        data_prefix=dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'), backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='NuScenesMetric', data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl', metric='bbox', backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP', rule='greater', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='Det3DVisualizationHook'))

default_scope = 'mmdet3d'
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False 