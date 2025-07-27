#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datetime import datetime
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules
from mmengine.dataset import DefaultSampler
from model_clip_with_bevfusion_infonce_distill_all import BEVFusionCLIPDistiller

# Register all modules
register_all_modules()

# Import all datasets
from mmdet3d.datasets import *



def build_model_from_cfg(config_path, checkpoint=None, device='cuda'):
    # Load config file
    cfg = Config.fromfile(config_path)

    # Build model using registry
    model = MODELS.build(cfg.model)

    # Load checkpoint if provided
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)

    # Move to device
    model = model.to(device)
    return model

def build_dataset_from_cfg(config_path, batch_size):
    """Build nuScenes dataset from config file"""
    print(f"Building dataset from config: {config_path}")
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build dataset
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # Custom collate function for mmdet3d data
    def custom_collate_fn(batch):
        return batch  # Return list of Det3DDataSample objects
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    print(f" Built dataloader with {len(train_dataset)} samples")
    return dataloader

def main():
    # Hardcode the configs and parameters instead of parsing from CLI
    teacher_config = "configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    student_config = "configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py"
    teacher_checkpoint = "checkpoints/epoch_20.pth"
    work_dir = "work_dirs/all_distill_run"
    batch_size = 8
    epochs = 20
    lr = 1e-4
    
    # Create work directory and setup logging
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'checkpoints'), exist_ok=True)
    
    # Initialize loss tracking
    loss_history = {
        'epoch_losses': [],
        'batch_losses': [],
        'training_config': {
            'teacher_config': teacher_config,
            'student_config': student_config,
            'teacher_checkpoint': teacher_checkpoint,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'start_time': datetime.now().isoformat()
        }
    }

    # 1. Load Teacher & Student
    print(f"Loading teacher model from: {teacher_config}")
    teacher_model = build_model_from_cfg(teacher_config, teacher_checkpoint, device='cuda')
    teacher_model.eval()

    print(f"Loading student model from: {student_config}")
    student_model = build_model_from_cfg(student_config, checkpoint=None, device='cuda')
    student_model.train()

    # 2. Create Distiller
    distiller = BEVFusionCLIPDistiller(teacher_model, student_model).cuda()

    # 3. Optimizer (only for student parameters)
    optimizer = torch.optim.Adam(distiller.student.parameters(), lr=lr)

    # 4. Build nuScenes Dataset from student config
    dataloader = build_dataset_from_cfg(student_config, batch_size=batch_size)

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, data_batch in enumerate(dataloader):
            # data_batch is now a list of Det3DDataSample objects
                        # Extract properly
            if isinstance(data_batch[0], dict):
                batch_inputs_dict = {'points': [sample['inputs']['points'] for sample in data_batch]}
                data_samples = [sample['data_samples'] for sample in data_batch]
            else:
                # Direct Det3DDataSample objects
                batch_inputs_dict = {'points': [sample.inputs['points'] for sample in data_batch]}
                data_samples = data_batch
            
            # Move data to CUDA
            if 'points' in batch_inputs_dict:
                points = batch_inputs_dict['points']
                if isinstance(points, list):
                    batch_inputs_dict['points'] = [p.cuda() for p in points]
                else:
                    batch_inputs_dict['points'] = points.cuda()

            if 'imgs' in batch_inputs_dict and batch_inputs_dict['imgs'] is not None:
                batch_inputs_dict['imgs'] = batch_inputs_dict['imgs'].cuda()

            optimizer.zero_grad()

            # Forward pass
            losses = distiller(batch_inputs_dict, data_samples)
            total_loss = losses["total_loss"]
            total_loss.backward()
            optimizer.step()

            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            
            # Log batch loss
            loss_history['batch_losses'].append({
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss': batch_loss,
                'timestamp': time.time()
            })
            
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {batch_loss:.4f}, Elapsed: {elapsed_time:.1f}s")

        # Calculate epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch loss
        epoch_info = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'total_loss': epoch_loss,
            'num_batches': len(dataloader),
            'epoch_time': epoch_time,
            'learning_rate': lr,
            'timestamp': time.time()
        }
        loss_history['epoch_losses'].append(epoch_info)
        
        print(f"Epoch [{epoch+1}/{epochs}] COMPLETE - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
        
                # Save checkpoint every epoch (official mmdetection3d format)
        checkpoint_path = os.path.join(work_dir, 'checkpoints', f'student_epoch_{epoch+1}.pth')
        checkpoint_data = {
            'state_dict': student_model.state_dict(),
            'meta': {
                'epoch': epoch + 1,
                'iter': batch_idx,
                'config': {
                    'teacher_config': teacher_config,
                    'student_config': student_config,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save loss history every epoch
        loss_json_path = os.path.join(work_dir, 'loss_history.json')
        with open(loss_json_path, 'w') as f:
            json.dump(loss_history, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    loss_history['training_config']['end_time'] = datetime.now().isoformat()
    loss_history['training_config']['total_training_time'] = total_time
    
    print(f"\ Training Complete!")
    print(f" Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f" Final Loss: {loss_history['epoch_losses'][-1]['avg_loss']:.4f}")
    
    # 6. Save Final Student Model
    final_model_path = os.path.join(work_dir, 'student_model_final.pth')
    final_checkpoint = {
        'state_dict': student_model.state_dict(),
        'meta': {
            'epoch': epochs,
            'final_loss': loss_history['epoch_losses'][-1]['avg_loss'],
            'epochs_trained': epochs,
            'config': loss_history['training_config']
        }
    }
    torch.save(final_checkpoint, final_model_path)
    
    # Save final loss history
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"💾 Final model saved: {final_model_path}")
    print(f"📝 Loss history saved: {loss_json_path}")
    print(f"📁 All files saved to: {work_dir}")

if __name__ == "__main__":
    main()