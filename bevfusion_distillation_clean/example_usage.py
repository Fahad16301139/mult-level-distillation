#!/usr/bin/env python3
"""
Example usage of BEVFusion Multi-Level Knowledge Distillation
"""

import torch
from model_clip_with_bevfusion_infonce_distill_all import (
    CLIP, 
    build_model,
    detect_dimensions_from_config
)

def example_basic_usage():
    """Basic example of how to use the distillation"""
    print("🚀 BEVFusion Distillation Example")
    print("=" * 50)
    
    # 1. Load configurations
    print("1️⃣ Loading configurations...")
    teacher_channels = detect_dimensions_from_config('teacher_config.py')
    student_channels = detect_dimensions_from_config('student_config.py')
    
    print(f"   Teacher channels: {teacher_channels}")
    print(f"   Student channels: {student_channels}")
    
    # 2. Create CLIP model
    print("\n2️⃣ Creating CLIP model...")
    clip_model = CLIP(
        embed_dim=512,
        teacher_channels=teacher_channels.get('pts_neck', 512),
        student_channels=student_channels.get('pts_neck', 256)
    )
    print("   ✅ CLIP model created")
    
    # 3. Example forward pass
    print("\n3️⃣ Example forward pass...")
    
    # Simulate teacher features (4 levels)
    teacher_features = {
        'voxel_encoder': torch.randn(1000, 10, 5),      # [N_voxels, feat1, feat2]
        'middle_encoder': torch.randn(2, 256, 64, 64),  # [B, C, H, W]
        'backbone': torch.randn(2, 384, 64, 64),        # [B, C, H, W]
        'neck': torch.randn(2, 512, 64, 64)             # [B, C, H, W]
    }
    
    # Simulate student features (4 levels)
    student_features = {
        'voxel_encoder': torch.randn(800, 10, 5),       # [N_voxels, feat1, feat2]
        'middle_encoder': torch.randn(2, 256, 32, 32),  # [B, C, H, W]
        'backbone': torch.randn(2, 192, 32, 32),        # [B, C, H, W]
        'neck': torch.randn(2, 256, 32, 32)             # [B, C, H, W]
    }
    
    # Forward pass
    result = clip_model(teacher_features, student_features)
    
    print(f"   ✅ Forward pass successful")
    print(f"   📊 Loss: {result['clip_total_loss'].item():.4f}")
    print(f"   📊 Levels processed: {result['processed_levels']}")
    print(f"   📊 Embedding dimension: {result['embedding_dim']}")
    
    return result

def example_training_loop():
    """Example of how to integrate into training loop"""
    print("\n🔄 Example Training Loop Integration")
    print("=" * 50)
    
    # This shows the pattern used in training_for_clip_infonce.py
    
    # 1. Create models (simplified)
    print("1️⃣ Create teacher and student models...")
    # teacher_model = build_model('teacher_config.py', 'teacher_checkpoint.pth')
    # student_model = build_model('student_config.py', None)
    print("   ✅ Models created (commented out for example)")
    
    # 2. Create CLIP model
    print("\n2️⃣ Create CLIP model...")
    clip_model = CLIP(embed_dim=512, teacher_channels=512, student_channels=256)
    print("   ✅ CLIP model created")
    
    # 3. Training loop pattern
    print("\n3️⃣ Training loop pattern...")
    print("   for epoch in range(num_epochs):")
    print("       for batch in dataloader:")
    print("           # Extract features from teacher and student")
    print("           teacher_features = extract_features(teacher_model, batch)")
    print("           student_features = extract_features(student_model, batch)")
    print("           ")
    print("           # Compute distillation loss")
    print("           loss_dict = clip_model(teacher_features, student_features)")
    print("           loss = loss_dict['clip_total_loss']")
    print("           ")
    print("           # Backward pass")
    print("           loss.backward()")
    print("           optimizer.step()")
    
    print("\n   ✅ Training loop pattern shown")

def example_checkpoint_usage():
    """Example of checkpoint saving/loading"""
    print("\n💾 Example Checkpoint Usage")
    print("=" * 50)
    
    # Simulate a trained student model
    print("1️⃣ Simulate trained student model...")
    # student_model = build_model('student_config.py', None)
    print("   ✅ Student model created (commented out for example)")
    
    # 2. Save checkpoint (official format)
    print("\n2️⃣ Save checkpoint in official format...")
    checkpoint = {
        'state_dict': {},  # student_model.state_dict()
        'epoch': 10,
        'meta': {
            'distillation_loss': 1.234,
            'approach': 'combined_multilevel_infonce_distillation'
        }
    }
    
    # torch.save(checkpoint, 'student_epoch_10.pth')
    print("   ✅ Checkpoint saved (commented out for example)")
    
    # 3. Load checkpoint
    print("\n3️⃣ Load checkpoint...")
    # loaded_checkpoint = torch.load('student_epoch_10.pth')
    # student_model.load_state_dict(loaded_checkpoint['state_dict'])
    print("   ✅ Checkpoint loaded (commented out for example)")
    
    # 4. Test with official tools
    print("\n4️⃣ Test with official BEVFusion tools...")
    print("   python tools/test.py student_config.py student_epoch_10.pth")
    print("   ✅ Compatible with official evaluation pipeline")

def main():
    """Run all examples"""
    try:
        # Run examples
        example_basic_usage()
        example_training_loop()
        example_checkpoint_usage()
        
        print("\n" + "=" * 50)
        print("🎉 All examples completed successfully!")
        print("\n📝 To run actual training:")
        print("1. python test_setup.py")
        print("2. Update paths in training_for_clip_infonce.py")
        print("3. python training_for_clip_infonce.py")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 