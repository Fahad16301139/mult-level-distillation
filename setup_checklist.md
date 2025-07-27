# CLIP Distillation Setup Checklist

## Required Files

### 1. Teacher Model Files
- **Config file**: `configs/teacher_bevfusion.py`
  - Path to your teacher model configuration
  - Example: BEVFusion with LiDAR+Camera
- **Checkpoint file**: `checkpoints/teacher_bevfusion.pth`
  - Pre-trained teacher model weights
  - Should be a fully trained, high-performance model

### 2. Student Model Files  
- **Config file**: `configs/student_bevfusion.py`
  - Path to your student model configuration
  - Usually smaller/faster than teacher
  - Example: BEVFusion with only LiDAR
- **Checkpoint file**: `checkpoints/student_bevfusion.pth` (optional)
  - Leave as `None` to start from scratch
  - Or provide path to resume training

### 3. Your Files Structure Should Look Like:
```
your_project/
├── configs/
│   ├── teacher_bevfusion.py        # Teacher config
│   └── student_bevfusion.py        # Student config
├── checkpoints/
│   ├── teacher_bevfusion.pth       # Teacher weights (required)
│   └── student_bevfusion.pth       # Student weights (optional)
├── training_for_clip.py            # Your CLIP trainer
├── model_clip_with_bevfusion.py    # Your CLIP model
└── main_clip_training.py           # Main script
```

## How to Get These Files

### Teacher Config & Checkpoint
1. Download from BEVFusion official repository
2. Or use your own trained BEVFusion model
3. Make sure it's a high-quality, fully-trained model

### Student Config
1. Create a lighter version of teacher config
2. Reduce model size (channels, layers, etc.)
3. Or use different modalities (LiDAR-only vs LiDAR+Camera)

## Usage Example

```python
# In main_clip_training.py
trainer = CLIPDistillationTrainer(
    teacher_config_path="configs/teacher_bevfusion.py",
    teacher_checkpoint_path="checkpoints/teacher_bevfusion.pth",     # Must exist
    student_config_path="configs/student_bevfusion.py", 
    student_checkpoint_path=None,  # None = start from scratch
    device='cuda'
)
```

## Common Issues

1. **FileNotFoundError**: Check if all paths exist
2. **ImportError**: Make sure mmdet3d is installed
3. **CUDA errors**: Check GPU memory and device availability
4. **Config errors**: Ensure configs are valid BEVFusion configs

## Quick Test

Run this to test your setup:
```bash
python main_clip_training.py
```

If successful, you should see:
```
DEBUG: Teacher config: configs/teacher_bevfusion.py
DEBUG: Teacher checkpoint: checkpoints/teacher_bevfusion.pth  
DEBUG: Loading Teacher model
DEBUG: Loading Teacher weights from checkpoints/teacher_bevfusion.pth
DEBUG: Teacher model loaded successfully
Setup completed successfully!
``` 