# BEVFusion Multi-Level Knowledge Distillation

This repository implements **multi-level knowledge distillation** for BEVFusion using **InfoNCE contrastive learning**. The approach distills knowledge from a large teacher model to a lightweight student model across all 4 levels of the BEVFusion pipeline.

## 🎯 Key Features

- **4-Level Distillation**: Distills knowledge from `voxel_encoder`, `middle_encoder`, `backbone`, and `neck` layers
- **Combined-Only InfoNCE**: Single contrastive loss on concatenated embeddings from all levels
- **One-Directional Learning**: Student learns from teacher (no teacher→student projection)
- **Official BEVFusion Compatible**: Checkpoints work with `tools/test.py`
- **Memory Efficient**: Adaptive pooling and gradient management

## 📁 Repository Structure

```
bevfusion_distillation_clean/
├── model_clip_with_bevfusion_infonce_distill_all.py  # Main CLIP model & distillation logic
├── training_for_clip_infonce.py                      # Training script
├── teacher_config.py                                 # Teacher model configuration
├── student_config.py                                 # Student model configuration
└── README.md                                         # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install MMDetection3D and dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision
pip install mmengine mmdet mmdet3d
```

### 2. Prepare Models

```bash
# Download teacher checkpoint (or use your own)
# Place teacher checkpoint at: work_dirs/teacher/epoch_20.pth

# Update paths in training_for_clip_infonce.py:
teacher_config = 'teacher_config.py'
teacher_checkpoint = 'work_dirs/teacher/epoch_20.pth'
student_config = 'student_config.py'
```

### 3. Run Training

```bash
python training_for_clip_infonce.py
```

### 4. Test Distilled Model

```bash
# Test with official BEVFusion tools
python tools/test.py student_config.py work_dirs/distillation_checkpoints/latest.pth
```

## 🔧 Architecture

### Multi-Level Feature Extraction

```python
# Extract from all 4 BEVFusion levels:
teacher_features = {
    'voxel_encoder': [N_voxels, 10, 5],      # Voxelized point features
    'middle_encoder': [B, 256, H, W],        # 2D BEV features
    'backbone': [B, 384, H, W],              # CNN backbone features  
    'neck': [B, 512, H, W]                   # Feature pyramid features
}
```

### Combined-Only InfoNCE Approach

```python
# 1. Pool each level to [B, C] embeddings
# 2. Concatenate all levels: [B, total_channels]
# 3. Single InfoNCE on complete pipeline representation
# 4. One-directional: student learns from teacher

teacher_combined = concat(voxel_emb, middle_emb, backbone_emb, neck_emb)
student_combined = concat(voxel_emb, middle_emb, backbone_emb, neck_emb)
loss = InfoNCE(student_combined, teacher_combined)
```

## 📊 Model Configurations

### Teacher Model (Large)
- **Voxel Size**: 0.075m
- **Backbone**: ResNet-50
- **Neck**: FPN with 512 channels
- **Parameters**: ~50M

### Student Model (Lightweight)  
- **Voxel Size**: 0.1m
- **Backbone**: ResNet-18
- **Neck**: FPN with 256 channels
- **Parameters**: ~15M

## 🎯 Training Details

### Loss Function
- **InfoNCE Loss**: Contrastive learning between teacher and student embeddings
- **Temperature**: 0.07 (CLIP-style)
- **Direction**: Student → Teacher only

### Optimization
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Gradient Clipping**: 0.5
- **Batch Size**: 4 samples

### Checkpoint Format
```python
{
    'state_dict': student_model.state_dict(),  # Compatible with tools/test.py
    'epoch': epoch_number,
    'meta': {
        'distillation_loss': avg_loss,
        'approach': 'combined_multilevel_infonce_distillation'
    }
}
```

## 🔍 Key Implementation Details

### 1. Voxel Encoder Handling
```python
# Special handling for voxel features [N_voxels, feat1, feat2]
# Convert to batch format: [B, combined_features]
voxel_features = voxel_features.flatten(1)  # [N_voxels, feat1*feat2]
pooled = voxel_features.mean(dim=0)         # [feat1*feat2]
batch_pooled = pooled.repeat(batch_size, 1) # [B, feat1*feat2]
```

### 2. Multi-Scale Backbone
```python
# Handle backbone with multiple scales
scales = [feat1, feat2]  # Different spatial resolutions
# Resize all to largest scale, then concatenate channels
resized = [interpolate(feat, target_size) for feat in scales]
combined = torch.cat(resized, dim=1)
```

### 3. Memory Management
```python
# Clear GPU cache before each extraction
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Use hooks for efficient feature extraction
# No unnecessary model copies
```

## 📈 Results

The distilled student model achieves:
- **Compression Ratio**: ~3x smaller than teacher
- **Performance**: Maintains competitive accuracy
- **Speed**: Faster inference due to smaller architecture
- **Memory**: Reduced memory footprint

## 🛠️ Customization

### Modify Model Configs
Edit `teacher_config.py` and `student_config.py` to use different architectures.

### Change Distillation Levels
Modify the level mappings in `extract_features()`:
```python
level_mappings = [
    ('voxel_encoder', 'teacher_voxel_encoder', 'student_voxel_encoder'),
    ('middle_encoder', 'teacher_middle_encoder', 'student_middle_encoder'),
    ('backbone', 'teacher_backbone', 'student_backbone'), 
    ('neck', 'teacher_neck', 'student_neck')
]
```

### Adjust Training Parameters
Modify in `training_for_clip_infonce.py`:
```python
num_epochs = 20
learning_rate = 0.001
batch_size = 4
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Errors**
   - Reduce batch size
   - Clear GPU cache more frequently
   - Use gradient checkpointing

2. **Voxel Encoder Not Found**
   - Check if `pts_voxel_encoder` exists in model
   - Fallback to `pts_voxel_layer` hook

3. **Checkpoint Loading Issues**
   - Ensure checkpoint format matches expected structure
   - Check device placement (CPU vs GPU)

### Debug Information
The training script provides extensive debug output:
- Feature extraction status
- Memory usage
- Loss values
- Checkpoint saving

## 📚 References

- **BEVFusion**: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation
- **InfoNCE**: Representation Learning with Contrastive Predictive Coding
- **CLIP**: Learning Transferable Visual Representations

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or issues, please open an issue on GitHub. 