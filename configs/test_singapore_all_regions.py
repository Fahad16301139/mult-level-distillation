_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']

# Test on ALL Singapore regions for comprehensive evaluation
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='NuScenesDataset',
                ann_file='nuscenes_infos_val_singapore_queenstown.pkl',  # Queenstown
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
                    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
                    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, 
                         pad_empty_sweeps=True, remove_close=True),
                    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
                    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
                         meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats']),
                ],
                test_mode=True),
            # Add other Singapore regions if you have them
            # dict(
            #     type='NuScenesDataset',
            #     ann_file='nuscenes_infos_val_singapore_onenorth.pkl',  # One North (if available as test)
            #     ... same config as above
            # ),
        ]
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val_singapore_queenstown.pkl',  # Main annotation file
    metric='bbox') 