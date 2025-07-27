#!/usr/bin/env python3

"""
Stage 2: Detection Fine-tuning - FULL MODEL VERSION
Comparison version: Fine-tune ALL parameters instead of freezing

EXPERIMENT: Compare two approaches:
- Version A (minimal_fix): Freeze all except bbox_head  
- Version B (this script): Fine-tune everything with low LR
"""

import torch
import torch.nn as nn
from mmengine.config import Config
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.utils import register_all_modules
from mmengine.runner import Runner
import os
from datetime import datetime

# Register all modules
register_all_modules()

def main():
    """Main function - full model fine-tuning for comparison"""
    print("🔥 Stage 2: Detection Fine-tuning - FULL MODEL VERSION")
    print("=" * 60)
    print("🧪 EXPERIMENT: Fine-tune ALL parameters (vs freeze-all-except-head)")
    print("=" * 60)
    
    # Configuration - Updated paths
    config_path = 'configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py'
    stage1_checkpoint_path = 'work_dirs/all_distill_run/student_model_final.pth'  # Use our completed distillation checkpoint
    
    print(f"📋 Config: {config_path}")
    print(f"🔄 Stage 1 checkpoint: {stage1_checkpoint_path}")
    
    # Load configuration
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = MODELS.build(cfg.model)
    model = model.cuda()
    print("✅ Model built and moved to GPU")
    
    # Load Stage 1 checkpoint with prefix handling
    print(f"\n🔄 Loading Stage 1 checkpoint: {stage1_checkpoint_path}")
    stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location='cuda')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in stage1_checkpoint:
        checkpoint_state_dict = stage1_checkpoint['model_state_dict']
    elif 'state_dict' in stage1_checkpoint:
        checkpoint_state_dict = stage1_checkpoint['state_dict']
    else:
        checkpoint_state_dict = stage1_checkpoint
    
    # Remove 'pts_' prefix from checkpoint keys to match model
    model_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith('pts_'):
            new_key = key[4:]  # Remove 'pts_' prefix
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    
    # Load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    print(f"✅ Checkpoint loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    
    # 🔧 DIFFERENT STRATEGY: Fine-tune ALL parameters (not just bbox_head)
    print("\n🔥 Applying FULL MODEL fine-tuning strategy...")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # Fine-tune EVERYTHING - no freezing
        param.requires_grad = True
        trainable_params += param.numel()
        
        # Log component breakdown
        if 'pts_neck' in name:
            print(f"  🔥 TRAINABLE (distilled neck): {name}")
        elif 'bbox_head' in name:
            print(f"  🔥 TRAINABLE (detection head): {name}")
        elif 'pts_backbone' in name:
            print(f"  🔥 TRAINABLE (backbone): {name}")
        elif 'pts_middle_encoder' in name:
            print(f"  🔥 TRAINABLE (middle encoder): {name}")
    
    total_params = frozen_params + trainable_params
    print(f"\n📊 Parameter Summary:")
    print(f"   🔥 Trainable: {trainable_params:,} ({100.0:.1f}%)")
    print(f"   ❄️ Frozen: {frozen_params:,} ({0.0:.1f}%)")
    print(f"   📊 Total: {total_params:,}")
    
    # Build dataset
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"✅ Dataset built with {len(train_dataset)} samples")
    
    # Create dataloader  
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # Same high performance batch size
        shuffle=True,
        num_workers=4,  # Same optimized workers
        collate_fn=lambda x: x,
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer for ALL parameters - VERY conservative learning rate
    all_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n🔧 Optimizer setup: {len(all_params)} parameter groups for FULL MODEL")
    
    # 🎯 KEY: Very conservative learning rate to preserve distilled knowledge
    optimizer = torch.optim.AdamW(
        all_params, 
        lr=1e-4,  # Much lower than 1e-4 to preserve distillation
        weight_decay=1e-4
    )
    
    print(f"✅ Optimizer configured for {sum(p.numel() for p in all_params):,} trainable parameters")
    print(f"🎯 Using conservative LR=5e-5 to preserve distilled knowledge")
    
    # Training loop (same as minimal fix)
    for epoch in range(20):
        print(f"\n--- Epoch {epoch+1}/20 ---")
        epoch_loss = 0.0
        successful_batches = 0
        
        for batch_idx, data_batch in enumerate(train_dataloader):
            # No batch limit - train on full dataset
                
            try:
                if isinstance(data_batch, list) and len(data_batch) > 0:
                    sample = data_batch[0]
                    
                    if not isinstance(sample, dict) or 'inputs' not in sample:
                        continue
                    
                    batch_inputs_dict = sample['inputs']
                    batch_data_samples = sample.get('data_samples', [])
                    
                    if not isinstance(batch_data_samples, list):
                        batch_data_samples = [batch_data_samples]
                    
                    # Move to GPU - EXACT same as working script
                    for key, value in batch_inputs_dict.items():
                        if isinstance(value, torch.Tensor):
                            batch_inputs_dict[key] = value.cuda()
                        elif isinstance(value, list):
                            for j, item in enumerate(value):
                                if isinstance(item, torch.Tensor):
                                    value[j] = item.cuda()
                    
                    if 'points' in batch_inputs_dict:
                        points = batch_inputs_dict['points']
                        if isinstance(points, torch.Tensor):
                            batch_inputs_dict['points'] = [points.cuda()]
                    
                    # Move data_samples to GPU - EXACT same as working script
                    for sample_item in batch_data_samples:
                        if hasattr(sample_item, 'gt_instances_3d') and sample_item.gt_instances_3d is not None:
                            gt_instances = sample_item.gt_instances_3d
                            for attr_name in dir(gt_instances):
                                if not attr_name.startswith('_'):
                                    try:
                                        attr_value = getattr(gt_instances, attr_name)
                                        if isinstance(attr_value, torch.Tensor):
                                            setattr(gt_instances, attr_name, attr_value.cuda())
                                    except:
                                        pass
                    
                    # Forward pass - EXACT same as working script
                    optimizer.zero_grad()
                    loss_dict = model.loss(batch_inputs_dict, batch_data_samples)
                    
                    if isinstance(loss_dict, dict):
                        total_loss = sum(loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor))
                    else:
                        total_loss = loss_dict
                    
                    if total_loss > 0:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ CRITICAL FIX!
                        optimizer.step()
                        
                        epoch_loss += total_loss.item()
                        successful_batches += 1
                        
                        if batch_idx % 50 == 0:
                            print(f"   Batch {batch_idx}: Loss = {total_loss.item():.4f}")
                    
            except Exception as e:
                if batch_idx % 50 == 0:
                    print(f"   ⚠️ Batch {batch_idx} failed: {str(e)[:100]}")
                continue
        
        avg_loss = epoch_loss / max(successful_batches, 1)
        print(f"✅ Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"work_dirs/stage2_full_finetune_epoch_{epoch+1}.pth"
            torch.save({
                'state_dict': model.state_dict(),  # ✅ Correct format
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'meta': {
                    'stage': 'detection_finetuning',
                    'method': 'full_finetune',
                    'config': config_path
                }
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = "work_dirs/stage2_full_finetune_final.pth"
    torch.save({
        'state_dict': model.state_dict(),  # ✅ Correct format
        'optimizer': optimizer.state_dict(),
        'epoch': 20,
        'training_completed': True,
        'meta': {
            'stage': 'detection_finetuning',
            'method': 'full_finetune',
            'config': config_path,
            'distillation_checkpoint': stage1_checkpoint_path
        }
    }, final_checkpoint)
    
    print(f"\n🎉 Training completed! Final model saved as: {final_checkpoint}")
    print("\n🧪 STAGE 2 DETECTION FINE-TUNING COMPLETED:")
    print(f"   ✅ Distillation checkpoint: {stage1_checkpoint_path}")
    print(f"   ✅ Final finetuned model: {final_checkpoint}")
    print("\n🔬 Ready for testing with the final model!")

if __name__ == "__main__":
    main() 