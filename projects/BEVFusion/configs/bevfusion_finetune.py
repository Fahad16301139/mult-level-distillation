_base_ = ['./bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# Load from the trained checkpoint
load_from = 'work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth'

# Fine-tuning learning rate (lower than the original)
optim_wrapper = dict(
    optimizer=dict(lr=0.00001)  # 10x smaller learning rate for fine-tuning
)

# Use a shorter training schedule for fine-tuning
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5,  # Reduce number of epochs for fine-tuning
    val_interval=1,
)

# Custom hooks
custom_hooks = [
    dict(type='DisableObjectSampleHook', disable_after_epoch=3),
]

# Smaller batch size if needed (depending on your GPU memory)
train_dataloader = dict(batch_size=2)  # Adjust based on your GPU memory

# Set resume to False to start fine-tuning from the pre-trained model
resume = False

# Reduce checkpoint saving frequency
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
) 