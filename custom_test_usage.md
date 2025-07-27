# Custom Test Script Usage Guide

## Overview

The `custom_test.py` script solves the checkpoint format and evaluation issues you've been experiencing, while calculating the **exact same metrics** as the original `tools/test.py`.

## Key Features

### ✅ **Automatic Checkpoint Format Fixing**
- Detects and fixes `'model_state_dict'` vs `'state_dict'` format mismatches
- Removes problematic key prefixes (`'model.'`, `'module.'`)
- Handles distillation checkpoints with `'student_model'` keys
- Creates temporary fixed checkpoints automatically

### ✅ **Cross-Region Evaluation Support**
- **Full dataset**: Uses complete NuScenes train/val sets
- **Singapore**: Uses Singapore Queenstown subset
- **Boston**: Uses Boston Seaport subset  
- **Original**: Uses whatever is in your config

### ✅ **Same Metrics as tools/test.py**
- **NDS** (NuScenes Detection Score)
- **mAP** (mean Average Precision)
- **Per-class mAP** for all 10 classes
- **mATE, mASE, mAOE** (Translation, Scale, Orientation errors)

## Basic Usage

```bash
# Basic evaluation with your checkpoint
python custom_test.py configs/your_config.py your_checkpoint.pth

# Evaluation with automatic format fixing
python custom_test.py configs/your_config.py your_checkpoint.pth --fix-checkpoint --verbose

# Cross-region evaluation on full dataset
python custom_test.py configs/your_config.py your_checkpoint.pth --dataset-type full
```

## Specific Use Cases

### 1. **Your Stage 2 Checkpoint** (Fixed Format)
```bash
python custom_test.py \
    configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py \
    stage2_detection_finetuning_fixed.pth \
    --dataset-type full \
    --fix-checkpoint \
    --verbose
```

### 2. **Professor's Evaluation Setup** (Cross-Region)
```bash
python custom_test.py \
    configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py \
    your_checkpoint.pth \
    --dataset-type singapore \
    --score-thr 0.05
```

### 3. **Distillation Checkpoint** (InfoNCE)
```bash
python custom_test.py \
    configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py \
    infonce_distill_final.pth \
    --fix-checkpoint \
    --verbose
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dataset-type` | Dataset configuration | `full`, `singapore`, `boston`, `original` |
| `--fix-checkpoint` | Auto-fix checkpoint format | `--fix-checkpoint` |
| `--verbose` | Detailed logging | `--verbose` |
| `--score-thr` | Detection confidence threshold | `--score-thr 0.05` |
| `--work-dir` | Output directory | `--work-dir work_dirs/evaluation` |

## What This Solves

### ❌ **Previous Issues**
```
✗ 0 mAP, 0 NDS scores
✗ "unexpected key in source state_dict: model_state_dict"
✗ Missing keys for core model components
✗ Dataset mismatch between training and evaluation
✗ Checkpoint format incompatibility
```

### ✅ **Fixed Issues**
```
✓ Proper mAP/NDS calculation
✓ Automatic checkpoint format conversion
✓ No missing keys - all components loaded
✓ Flexible dataset configuration
✓ MMDetection3D compatible format
```

## Example Output

```
🚀 Custom BEVFusion Test Script
============================================================
📁 Config: configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py
💾 Checkpoint: stage2_detection_finetuning_fixed.pth
🌍 Dataset type: full
🎯 Score threshold: 0.1

🔧 Fixing checkpoint format...
✅ Fixed checkpoint saved to: /tmp/tmp_abc123/fixed_checkpoint.pth
   - Original parameters: 1247
   - Cleaned parameters: 1247

🏗️ Building runner...
✅ Runner created successfully

🎯 Starting evaluation...
📊 This will calculate NDS, mAP, and per-class metrics

🎉 Evaluation completed!
📈 Results Summary:
   mAP: 0.3456
   NDS: 0.4123
   mATE: 0.2345
   mASE: 0.1234
   mAOE: 0.0987

📊 Per-class Performance:
   car: 0.5234
   truck: 0.3456
   construction_vehicle: 0.2345
   bus: 0.4567
   trailer: 0.3123
   barrier: 0.6789
   motorcycle: 0.2456
   bicycle: 0.1789
   pedestrian: 0.3456
   traffic_cone: 0.4567

✅ Custom test completed successfully!
```

## Advantages Over tools/test.py

1. **Robust Checkpoint Loading**: Handles various checkpoint formats automatically
2. **Cross-Region Flexibility**: Easy dataset switching without config editing  
3. **Verbose Debugging**: Detailed information about what's happening
4. **Automatic Fixes**: No manual checkpoint preprocessing needed
5. **Same Accuracy**: Uses identical evaluation pipeline as tools/test.py

## Next Steps

1. **Run evaluation** with your fixed checkpoint
2. **Compare results** across different dataset configurations
3. **Use lower score thresholds** if needed (e.g., `--score-thr 0.05`)
4. **Check per-class performance** to identify weak areas

This script should give you the **exact same evaluation results** as tools/test.py, but with much better compatibility for your custom checkpoint formats! 🎯 