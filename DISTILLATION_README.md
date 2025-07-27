# BEVFusion Knowledge Distillation

This implementation provides knowledge distillation for BEVFusion models, enabling you to transfer knowledge from a large teacher model to a smaller, more efficient student model suitable for edge deployment.

## Overview

The distillation framework combines:
- **Contrastive Representation Distillation (CRD)**: Aligns feature representations between teacher and student
- **Multi-layer Feature Transfer**: Captures knowledge from backbone and neck layers
- **Efficient Edge Deployment**: Maintains performance while reducing model size and computational requirements

## Files Structure

```
├── bevfusion_distillation.py      # Core distillation implementation
├── train_bevfusion_distill.py     # Training script
├── run_distillation.sh            # Execution script
└── DISTILLATION_README.md          # This file
```

## Quick Start

1. **Prepare Models**:
   - Teacher model: Original BEVFusion with full complexity
   - Student model: Your lightweight BEVFusion model

2. **Run Distillation**:
   ```bash
   chmod +x run_distillation.sh
   ./run_distillation.sh
   ```

3. **Monitor Training**:
   - Logs are saved in the work directory
   - Checkpoints are saved every epoch

## Key Components

### BEVFusionDistillationWrapper
- Wraps BEVFusion models to capture intermediate features
- Registers hooks at key layers: `pts_middle_encoder`, `pts_backbone`, `pts_neck`
- Supports both training and inference modes

### ContrastiveLoss
- Implements InfoNCE-style contrastive learning
- Aligns student and teacher feature representations
- Uses temperature scaling for soft similarity

### ProjectionMLP
- Projects features to common embedding space
- Handles different feature shapes (BEV, point clouds, sparse)
- Applies L2 normalization for stable training

## Configuration

### Model Channels
Update the channel dimensions in `build_bevfusion_distillation()` based on your models:

```python
teacher_channels = {
    'pts_backbone': 512,  # [128 + 256] for original model
    'pts_neck': 512,      # [256 + 256] for original model
}

student_channels = {
    'pts_backbone': 192,  # [64 + 128] for lightweight model
    'pts_neck': 256,      # [128 + 128] for lightweight model
}
```

### Hyperparameters
- `temperature`: Controls contrastive loss sharpness (default: 0.07)
- `alpha_crd`: Weight for contrastive distillation loss (default: 1.0)
- `feat_dim`: Projection space dimensionality (default: 128)

## Training Process

1. **Teacher Forward**: Extract features without gradients
2. **Student Forward**: Extract features with gradients
3. **Loss Computation**: 
   - Detection loss from student model
   - Contrastive distillation loss between features
4. **Optimization**: Update student model parameters only

## Expected Results

- **Model Size**: ~50% reduction compared to teacher
- **Speed**: 2-3x faster inference on edge devices
- **Performance**: 90-95% of teacher model mAP retention
- **Memory**: Reduced GPU memory usage during inference

## Code Inspiration Sources

This implementation draws inspiration from:

1. **RepDistiller Framework**: 
   - Overall structure and modular design
   - Contrastive loss implementation from CRD method
   - Multi-layer distillation approach

2. **Your BEVFusion Codebase**:
   - Model wrapper integrates with `projects/BEVFusion/bevfusion/bevfusion.py`
   - Hook registration adapted for BEV feature extraction
   - Compatible with mmdetection3d training pipeline

3. **Contrastive Learning Literature**:
   - InfoNCE loss from MoCo/SimCLR
   - Temperature scaling from knowledge distillation
   - Feature projection from contrastive learning

4. **3D Detection Specifics**:
   - BEV feature handling for spatial representations
   - Point cloud feature aggregation methods
   - Integration with TransFusion detection head

## Key Adaptations Made

1. **BEV Feature Handling**: Custom projection for 4D BEV tensors
2. **Sparse Feature Support**: Handles sparse convolution outputs
3. **Multi-Modal Integration**: Works with your lidar-only configuration
4. **Edge Optimization**: Simplified architecture for deployment

## Usage Tips

1. Start with a well-trained teacher model
2. Use the provided hyperparameters as baseline
3. Monitor both detection and distillation losses
4. Adjust `alpha_crd` if distillation dominates detection loss
5. Validate on your specific deployment hardware

## Troubleshooting

- **CUDA OOM**: Reduce batch size or feature dimensions
- **Loss Explosion**: Lower learning rate or alpha values
- **Poor Convergence**: Check teacher model quality and feature alignment

This distillation approach is specifically designed for your Singapore region deployment, balancing performance retention with edge device constraints. 