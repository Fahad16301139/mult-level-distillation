# BEVFusion Knowledge Distillation Configuration
# Based on your lightweight Singapore config

_base_ = [
    'bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py'
]

# Distillation-specific configurations
distillation_cfg = dict(
    # Teacher model configuration
    teacher_config='configs/bevfusion_lidar_voxel075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
    teacher_checkpoint='checkpoints/bevfusion_original.pth',
    
    # Distillation method settings
    method='comprehensive',  # Use comprehensive distillation
    temperature=0.07,        # Contrastive loss temperature
    kd_temperature=4.0,      # Knowledge distillation temperature
    feat_dim=128,           # Feature projection dimension
    
    # Loss weights
    alpha_contrastive=1.0,   # Weight for contrastive loss
    alpha_attention=0.5,     # Weight for attention transfer
    alpha_kd=0.3,           # Weight for knowledge distillation
    
    # Feature layers to distill
    distill_layers=[
        'pts_middle_encoder',
        'pts_backbone', 
        'pts_neck'
    ],
    
    # Channel dimensions (extracted from your config)
    student_channels=dict(
        pts_middle_encoder=128,  # From your lightweight config
        pts_backbone=192,        # 64 + 128 from backbone out_channels
        pts_neck=256            # 128 + 128 from neck out_channels
    ),
    
    teacher_channels=dict(
        pts_middle_encoder=256,  # From original BEVFusion
        pts_backbone=384,        # 128 + 256 from original backbone
        pts_neck=512            # 256 + 256 from original neck
    )
)

# Override training settings for distillation
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=25,  # Slightly longer for distillation
    val_interval=3  # More frequent validation
)

# Optimizer settings for distillation
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=5e-5,  # Lower learning rate for distillation
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000  # Longer warmup for distillation
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        by_epoch=True,
        begin=0,
        end=25,
        eta_min_ratio=0.01,
        convert_to_iter_based=True
    )
]

# Custom hooks for distillation
custom_hooks = [
    dict(
        type='CheckpointHook',
        interval=5,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater',
        max_keep_ckpts=5  # Keep more checkpoints during distillation
    ),
    # Add custom distillation logging hook
    dict(
        type='DistillationLoggerHook',
        log_interval=50,
        log_distill_losses=True
    )
]

# Work directory
work_dir = 'work_dirs/bevfusion_distillation_singapore'

# Resume settings
resume = False
load_from = None

# Enable mixed precision training for efficiency
fp16 = dict(loss_scale=dict(init_scale=512)) 