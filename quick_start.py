#!/usr/bin/env python3
"""
Quick Start Script for BEVFusion with Multi-Level Knowledge Distillation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'mmcv', 'mmdet', 'mmengine', 'mmdet3d'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available. Training will be slower on CPU.")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_data_directory():
    """Check if data directory exists."""
    data_dir = Path("data/nuscenes")
    if data_dir.exists():
        print(f"✅ Data directory found: {data_dir}")
        return True
    else:
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   Please download the NuScenes dataset first.")
        return False

def create_sample_config():
    """Create a sample configuration for quick testing."""
    config_content = '''# Sample configuration for BEVFusion distillation
teacher_config = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
student_config = "projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
teacher_checkpoint = "path/to/your/teacher_checkpoint.pth"
work_dir = "./work_dirs/quick_start_distillation"
batch_size = 2
epochs = 5
'''
    
    with open("quick_start_config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Created quick_start_config.py")

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test basic imports
        from model_clip_with_bevfusion_infonce_distill_all import CLIPBEVFusionInfoNCEDistillation
        print("✅ Distillation model import successful")
        
        # Test configuration loading
        from mmengine.config import Config
        config = Config.fromfile("projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
        print("✅ Configuration loading successful")
        
        print("✅ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick Start for BEVFusion Distillation")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration")
    parser.add_argument("--run-test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    print("🚀 BEVFusion with Multi-Level Knowledge Distillation - Quick Start")
    print("=" * 70)
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check CUDA
        check_cuda()
        
        # Check data directory
        check_data_directory()
    
    if args.create_config:
        create_sample_config()
    
    if args.run_test:
        if not run_quick_test():
            sys.exit(1)
    
    print("\n📚 Next Steps:")
    print("1. Download NuScenes dataset: python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes")
    print("2. Train teacher model: python tools/train.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    print("3. Run distillation: python training_for_clip_infonce.py --teacher-config <teacher_config> --student-config <student_config> --teacher-checkpoint <teacher_checkpoint>")
    print("4. Evaluate results: python tools/test.py <student_config> <checkpoint> --eval bbox")
    
    print("\n📖 For more information, see README.md")
    print("🐛 For troubleshooting, see the troubleshooting section in README.md")

if __name__ == "__main__":
    main() 