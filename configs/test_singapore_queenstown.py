_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']

# Override test_dataloader and test_evaluator explicitly to ensure Singapore Queenstown is used
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val_singapore_queenstown.pkl',  # FORCE Singapore Queenstown
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val_singapore_queenstown.pkl',  # FORCE Singapore Queenstown
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')

# Remove any train/val configurations to ensure only testing
train_dataloader = None
train_cfg = None
val_dataloader = None
val_evaluator = None 