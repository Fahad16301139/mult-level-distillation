# Simple BEVFusion Distillation Config
# This works with the standard tools/train.py script

# Use your lightweight student model as base
_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']

# Custom imports for distillation
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'],
    allow_failed_imports=False
)

# Work directory
work_dir = 'work_dirs/bevfusion_distillation_simple'

# Load pretrained teacher checkpoint
load_from = 'checkpoints/epoch_20.pth'

# Reduce training for testing
train_cfg = dict(max_epochs=2, type='EpochBasedTrainLoop', val_interval=1)

# Reduce batch size for memory
train_dataloader = dict(
    batch_size=1,  # Very small for testing
    dataset=dict(
        type='NuScenesDataset',
        ann_file='nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
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
        ],
        test_mode=False,
        use_valid_flag=True),
    num_workers=1,  # Reduced for stability
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

# Reduce validation batch size too
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
        pipeline=[
            dict(backend_args=None, coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
            dict(backend_args=None, load_dim=5, pad_empty_sweeps=True, remove_close=True, sweeps_num=9, type='LoadPointsFromMultiSweeps', use_dim=5),
            dict(point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], type='PointsRangeFilter'),
            dict(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'], meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats'], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
) 