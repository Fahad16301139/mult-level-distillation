# 🚀 How to Run MSE Distillation - Step by Step Guide

You have **3 options** for MSE distillation. Here's exactly what to do:

## 📁 **Option 1: Use New Simple Script (RECOMMENDED)**

**File**: `tutorial_mse_bevfusion.py` (just created)

**Steps**:
1. **Modify paths** in the script (lines 142-144):
   ```python
   teacher_config = 'YOUR_TEACHER_CONFIG.py'
   student_config = 'YOUR_STUDENT_CONFIG.py'  
   teacher_checkpoint = 'YOUR_TEACHER_CHECKPOINT.pth'
   ```

2. **Run it**:
   ```bash
   python tutorial_mse_bevfusion.py
   ```

**What it does**:
- ✅ Simple MSE loss (your method)
- ✅ Automatic shape alignment
- ✅ PyTorch tutorial structure
- ✅ Easy to understand

---

## 📁 **Option 2: Use Your Existing Clean Script**

**File**: `final_working_mse_distillation_clean.py` (already exists)

**Steps**:
1. **Modify paths** in the `main()` function (around line 640):
   ```python
   teacher_config_path = 'YOUR_TEACHER_CONFIG.py'
   student_config_path = 'YOUR_STUDENT_CONFIG.py'
   teacher_checkpoint_path = 'YOUR_TEACHER_CHECKPOINT.pth'
   ```

2. **Run it**:
   ```bash
   python final_working_mse_distillation_clean.py
   ```

**What it does**:
- ✅ Your full MSE implementation
- ✅ Complete logging
- ✅ JSON metrics
- ✅ Advanced features

---

## 📁 **Option 3: Use Your Simple Training Script**

**File**: `simple_mse_distillation.py` (already exists)

**Steps**:
1. **Check the config paths** inside the file
2. **Run it**:
   ```bash
   python simple_mse_distillation.py
   ```

---

## 🎯 **Which One to Choose?**

| Script | Best For | Complexity | Features |
|--------|----------|------------|----------|
| `tutorial_mse_bevfusion.py` | **Beginners** | Simple | Basic MSE + Tutorial methods |
| `final_working_mse_distillation_clean.py` | **Full training** | Advanced | Complete MSE + Logging |
| `simple_mse_distillation.py` | **Quick test** | Medium | Basic MSE |

## 🔧 **What You Need to Change**

**For ANY script**, you need to modify these paths:

1. **Teacher config**: Path to your big/teacher model config
2. **Student config**: Path to your small/student model config  
3. **Teacher checkpoint**: Path to trained teacher weights (.pth file)

**Example**:
```python
# Your teacher (big model)
teacher_config = 'work_dirs/bevfusion_teacher/config.py'
teacher_checkpoint = 'work_dirs/bevfusion_teacher/epoch_20.pth'

# Your student (small model)
student_config = 'configs/bevfusion_student_lightweight.py'
```

## 🎓 **Expected Output**

When you run any script, you should see:
```
🚀 MSE Distillation for BEVFusion
Loading teacher...
Teacher frozen: 8,306,030 params
Loading student...
Student trainable: 4,706,350 params

Epoch 1/20
  Batch 0: Loss = 0.024567
  Batch 10: Loss = 0.019234
  Average Loss: 0.021456

Epoch 2/20
  Batch 0: Loss = 0.018123
  ...
```

## ⚡ **Quick Start (Recommended)**

1. **Use the new simple script**:
   ```bash
   python tutorial_mse_bevfusion.py
   ```

2. **If it fails**, check these paths and fix them:
   - Teacher config path
   - Student config path
   - Teacher checkpoint path

3. **That's it!** The script will train your student model using MSE distillation.

## 🆘 **If You Get Errors**

**Common issues**:
- ❌ **Path not found**: Fix the config/checkpoint paths
- ❌ **CUDA error**: Add `device='cpu'` if no GPU
- ❌ **Import error**: Make sure you're in the right environment

**Quick fix**: Run the simplest script first:
```bash
python simple_mse_distillation.py
``` 