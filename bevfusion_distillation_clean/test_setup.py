#!/usr/bin/env python3
"""
Test script to verify BEVFusion distillation setup
"""

import torch
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from model_clip_with_bevfusion_infonce_distill_all import (
            CLIP, 
            BEVFusionCLIPWrapper, 
            build_model,
            detect_dimensions_from_config
        )
        print("✅ CLIP modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CLIP modules: {e}")
        return False
    
    try:
        from mmengine.config import Config
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        print("✅ MMDetection3D modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MMDetection3D modules: {e}")
        return False
    
    return True

def test_configs():
    """Test if config files can be loaded"""
    print("\n🔍 Testing config files...")
    
    try:
        from mmengine.config import Config
        
        # Test teacher config
        teacher_config = Config.fromfile('teacher_config.py')
        print("✅ Teacher config loaded successfully")
        
        # Test student config  
        student_config = Config.fromfile('student_config.py')
        print("✅ Student config loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load configs: {e}")
        return False

def test_clip_model():
    """Test CLIP model creation"""
    print("\n🔍 Testing CLIP model creation...")
    
    try:
        from model_clip_with_bevfusion_infonce_distill_all import CLIP
        
        # Create CLIP model with sample dimensions
        clip_model = CLIP(
            embed_dim=512,
            teacher_channels=512,
            student_channels=256
        )
        print("✅ CLIP model created successfully")
        
        # Test forward pass with dummy data
        dummy_teacher = {
            'neck': torch.randn(2, 512, 64, 64),
            'backbone': torch.randn(2, 384, 64, 64),
            'middle_encoder': torch.randn(2, 256, 64, 64),
            'voxel_encoder': torch.randn(1000, 10, 5)
        }
        
        dummy_student = {
            'neck': torch.randn(2, 256, 32, 32),
            'backbone': torch.randn(2, 192, 32, 32), 
            'middle_encoder': torch.randn(2, 256, 32, 32),
            'voxel_encoder': torch.randn(800, 10, 5)
        }
        
        result = clip_model(dummy_teacher, dummy_student)
        print(f"✅ CLIP forward pass successful, loss: {result['clip_total_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - training will use CPU (slow)")
        return True  # Not a failure, just slower

def main():
    """Run all tests"""
    print("🚀 BEVFusion Distillation Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configs, 
        test_clip_model,
        test_cuda
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for training.")
        print("\n📝 Next steps:")
        print("1. Update paths in training_for_clip_infonce.py")
        print("2. Download teacher checkpoint")
        print("3. Run: python training_for_clip_infonce.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 