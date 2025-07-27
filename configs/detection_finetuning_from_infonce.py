# Detection Fine-tuning Config: Load Pretrained Neck + Train BBox Head
# =====================================================================
# This config loads InfoNCE-distilled neck weights and trains only bbox head

_base_ = [
    'bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py'
]

# Load pretrained neck weights from InfoNCE checkpoint
load_from = 'work_dirs/infonce_distill_epoch_100.pth'

# Training configuration for bbox head only
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50,  # Fewer epochs for fine-tuning
    val_interval=5
)

# Lower learning rate for fine-tuning
lr = 5e-5  # Much lower than standard 1e-4

# Optimizer for bbox head only
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=1e-4),
    # Freeze all parameters except bbox head
    paramwise_cfg=dict(
        custom_keys={
            'pts_backbone': dict(lr_mult=0.0),  # Freeze backbone
            'pts_neck': dict(lr_mult=0.0),      # Freeze neck (pretrained)
            'pts_bbox_head': dict(lr_mult=1.0), # Train bbox head
        }
    )
)

# Learning rate scheduler
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=50,
    eta_min=1e-6,
    by_epoch=True
)

# Custom hook to handle weight loading
custom_hooks = [
    dict(
        type='CustomWeightLoadHook',
        priority=99,
        neck_only=True,  # Load only neck weights
        reinit_bbox_head=True  # Reinitialize bbox head
    )
]

# Training settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=True,
        save_best='auto'
    )
)

# Work directory
work_dir = 'work_dirs/detection_finetuning_from_infonce'

# Resume settings
resume = False
auto_resume = True 