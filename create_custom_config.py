#!/usr/bin/env python
import os
import argparse
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Create a custom config file for BEVFusion')
    parser.add_argument('--base-config', default='projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py', 
                        help='Base config file')
    parser.add_argument('--out-config', default='configs/bevfusion_lidar_mini.py',
                        help='Output config file path')
    parser.add_argument('--data-root', default='data/nuscenes/',
                        help='Data root path')
    parser.add_argument('--mini-ann', default='nuscenes_infos_val_mini.pkl',
                        help='Mini dataset annotation file name')
    parser.add_argument('--disable-eval', action='store_true',
                        help='Disable evaluation in the config')
    return parser.parse_args()

def create_custom_config():
    args = parse_args()
    
    # Load base config
    cfg = Config.fromfile(args.base_config)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out_config), exist_ok=True)
    
    # Ensure mini annotation file exists
    mini_ann_path = os.path.join(args.data_root, args.mini_ann)
    if not os.path.exists(mini_ann_path):
        print(f"Warning: Mini annotation file {mini_ann_path} does not exist.")
        print("You'll need to create it before using this config.")
    
    # Modify config for mini dataset
    cfg.data_root = args.data_root
    
    # Update val/test dataloader
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.dataset.ann_file = args.mini_ann
        cfg.val_dataloader.dataset.data_prefix = dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'
        )
    
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.ann_file = args.mini_ann
        cfg.test_dataloader.dataset.data_prefix = dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'
        )
    
    # Update evaluator
    if hasattr(cfg, 'val_evaluator'):
        cfg.val_evaluator.ann_file = os.path.join(args.data_root, args.mini_ann)
    
    if hasattr(cfg, 'test_evaluator'):
        cfg.test_evaluator.ann_file = os.path.join(args.data_root, args.mini_ann)
    
    # Disable evaluation if requested
    if args.disable_eval:
        cfg.val_evaluator = None
        cfg.test_evaluator = None
        print("Evaluation has been disabled in the config.")
    
    # Add a comment to indicate this is a custom config
    custom_config_str = f"""
# ===========================================================================
# Custom BEVFusion config for mini dataset
# Base config: {args.base_config}
# Created by create_custom_config.py
# ===========================================================================

"""
    
    # Dump the config to file
    config_str = custom_config_str + cfg.dump()
    with open(args.out_config, 'w') as f:
        f.write(config_str)
    
    print(f"Custom config created at: {args.out_config}")

if __name__ == '__main__':
    create_custom_config() 