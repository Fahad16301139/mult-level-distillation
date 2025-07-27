# Teacher Model - Full BEVFusion for Mini nuScenes
# Based on lightweight config but with full model parameters
_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']

# TEACHER MODEL: Reverse all lightweight reductions back to FULL model
model = dict(
    # Full voxel processing - TEACHER settings
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=10,  # TEACHER: 10 (vs student 5)
            max_voxels=[120000, 160000],  # TEACHER: Original large voxel count
            voxel_size=[0.075, 0.075, 0.2],  # TEACHER: Original smaller voxels
        )
    ),
    
    # Full sparse encoder
    pts_middle_encoder=dict(
        sparse_shape=[1440, 1440, 41],  # TEACHER: Original large sparse shape
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 256)),  # TEACHER: Original full channels with 256 final
    ),
    
    # Full backbone - TEACHER settings  
    pts_backbone=dict(
        in_channels=256,  # TEACHER: Match full middle encoder output
        out_channels=[128, 256],  # TEACHER: [128,256] (vs student [64,128])
        layer_nums=[5, 5],  # TEACHER: [5,5] (vs student [3,3])
    ),
    
    # Full neck - TEACHER settings
    pts_neck=dict(
        in_channels=[128, 256],  # TEACHER: Match backbone output
        out_channels=[256, 256],  # TEACHER: [256,256] (vs student [128,128])
    ),
    
    # Full TransFusion head - TEACHER settings
    bbox_head=dict(
        in_channels=512,  # TEACHER: Match full neck output (256+256)
        hidden_channel=128,  # TEACHER: 128 (vs student 64)
        num_proposals=200,   # TEACHER: 200 (vs student 100)
        num_decoder_layers=1,  # Keep at 1 for both
        decoder_layer=dict(
            cross_attn_cfg=dict(embed_dims=128, num_heads=8),  # TEACHER: 128 dims, 8 heads
            self_attn_cfg=dict(embed_dims=128, num_heads=8),   # TEACHER: 128 dims, 8 heads
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,  # TEACHER: 256 (vs student 128)
            ),
            pos_encoding_cfg=dict(num_pos_feats=128),  # TEACHER: 128 pos features
        ),
        train_cfg=dict(
            grid_size=[1440, 1440, 41],  # TEACHER: Match sparse shape
            voxel_size=[0.075, 0.075, 0.2],  # TEACHER: Match voxel size
        ),
        test_cfg=dict(
            grid_size=[1440, 1440, 41],  # TEACHER: Match sparse shape  
            voxel_size=[0.075, 0.075],  # TEACHER: Match voxel size XY
        ),
        bbox_coder=dict(
            voxel_size=[0.075, 0.075],  # TEACHER: Match voxel size XY
        ),
    ),
)

# TRAINING: Teacher-specific settings
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),  # TEACHER: Same LR as student base
)

# Work directory for teacher model
work_dir = 'work_dirs/bevfusion_teacher_full_mini'

# Training config
train_cfg = dict(
    max_epochs=20,  # TEACHER: Full training epochs
    val_interval=5,  # Same validation frequency
)

# TRAINING DATA: Mini nuScenes dataset
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5, truck=5, bus=5, trailer=5, construction_vehicle=5,
            traffic_cone=5, barrier=5, motorcycle=5, bicycle=5, pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
        barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

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
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectSample', db_sampler=db_sampler),
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
        load_dim=5,
        use_dim=5,
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

# DATASETS: Mini nuScenes
train_dataloader = dict(
    batch_size=2,  # TEACHER: Smaller batch for teacher training on mini dataset
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',  # Mini nuScenes train
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        use_valid_flag=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',  # Mini nuScenes val
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',  # Mini nuScenes val
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# LEARNING RATE SCHEDULER
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
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.8947368421052632,
        by_epoch=True,
        begin=0,
        end=8,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        by_epoch=True,
        begin=8,
        end=20,
        convert_to_iter_based=True)
]

# HOOKS
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater')
] 