# Student Model Configuration for Knowledge Distillation
# Based on bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py
# But with even more aggressive reductions for student model

_base_ = ['bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py']  # Student base config

# STUDENT MODEL: Even more aggressive reductions
model = dict(
    # Further reduce voxel processing
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=3,  # STUDENT: 3 (vs teacher 10, lightweight 5)
            max_voxels=[50000, 60000],  # STUDENT: 50% further reduction
            voxel_size=[0.15, 0.15, 0.2],  # STUDENT: Even larger voxels
        )
    ),
    
    # Smaller backbone for student
    pts_backbone=dict(
        out_channels=[32, 64],  # STUDENT: [32,64] (vs teacher [128,256], lightweight [64,128])
        layer_nums=[2, 2],  # STUDENT: [2,2] (vs teacher [5,5], lightweight [3,3])
    ),
    
    # Minimal TransFusion head
    bbox_head=dict(
        hidden_channel=32,  # STUDENT: 32 (vs teacher 128, lightweight 64)
        num_proposals=50,   # STUDENT: 50 (vs teacher 200, lightweight 100)
        num_decoder_layers=1,  # Keep at 1 layer
        decoder_layer=dict(
            cross_attn_cfg=dict(embed_dims=32, num_heads=2),  # STUDENT: 32 dims, 2 heads
            self_attn_cfg=dict(embed_dims=32, num_heads=2),   # STUDENT: 32 dims, 2 heads
            ffn_cfg=dict(
                embed_dims=32,
                feedforward_channels=64,  # STUDENT: 64 (vs teacher 256, lightweight 128)
            ),
            pos_encoding_cfg=dict(num_pos_feats=32),  # STUDENT: 32 pos features
        ),
        in_channels=128,  # Match neck output
    ),
    
    # Smaller neck
    pts_neck=dict(
        out_channels=[64, 64],  # STUDENT: [64,64] (vs teacher [256,256], lightweight [128,128])
    ),
)

# TRAINING: Knowledge Distillation specific settings
optim_wrapper = dict(
    optimizer=dict(lr=0.0002),  # STUDENT: Slightly higher LR for distillation
)

# Distillation-specific training config
train_cfg = dict(
    max_epochs=15,  # STUDENT: Shorter training with teacher guidance
    val_interval=3,  # More frequent validation
)

# Work directory for student model
work_dir = 'work_dirs/bevfusion_student_singapore_distillation'

# Custom hooks for distillation
custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater',
    ),
] 