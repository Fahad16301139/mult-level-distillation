import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# Import our CLIP components - FIXED TO MATCH WORKING SCRIPT
from model_clip_with_bevfusion_infonce_distill_all import (
    CLIP, 
    BEVFusionCLIPWrapper, 
    build_model,
    detect_dimensions_from_config
)


class SimpleTrainer:
    """Very simple distillation trainer - MATCHING WORKING SCRIPT INTERFACE"""
    
    def __init__(self, teacher_config, teacher_checkpoint, student_config, device='cuda:0'):
        self.device = device
        self.student_config = student_config  # Store for checkpoint saving
        
        # 🔥 AUTO-DETECTION: Extract neck dimensions from config files (LIKE WORKING SCRIPT)
        print("🔍 AUTO-DETECTING neck dimensions from config files...")
        teacher_channels = detect_dimensions_from_config(teacher_config)
        student_channels = detect_dimensions_from_config(student_config)
        
        print(f"✅ AUTO-DETECTED Teacher channels: {teacher_channels}")
        print(f"✅ AUTO-DETECTED Student channels: {student_channels}")
        
        # Store for later use
        self.teacher_channels = teacher_channels.get('pts_neck', 512)  # fallback to 512
        self.student_channels = student_channels.get('pts_neck', 256)  # fallback to 256
        
        print(f"🎯 Using Teacher: {self.teacher_channels} channels, Student: {self.student_channels} channels")
        
        # Build models
        teacher_model = self._build_model(teacher_config, teacher_checkpoint)
        student_model = self._build_model(student_config, None)
        
        # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        
        # 🔥 FIXED: Using DIRECT models like working script (NO WRAPPERS!)
        # self.teacher_wrapper = BEVFusionCLIPWrapper(teacher_model, is_teacher=True)  # REMOVED
        # self.student_wrapper = BEVFusionCLIPWrapper(student_model, is_teacher=False)  # REMOVED
        
        # Store models for DIRECT access (like working script)
        self.teacher_model = teacher_model
        self.student_model = student_model
        print("🔥 Using DIRECT model extraction (like working script) - NO WRAPPERS!")
        
        # Create CLIP and loss - FIXED TO MATCH WORKING SCRIPT
        self.clip_model = CLIP(
            embed_dim=512,
            teacher_channels=self.teacher_channels,
            student_channels=self.student_channels
        ).to(device)
        
        # InfoNCE loss is now calculated inside CLIP model using combined-only approach
        print(f"🔥 CLIP model created with AUTO-DETECTED dimensions: T={self.teacher_channels}, S={self.student_channels}")
        
        # Simple optimizer - just student model
        self.optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    def _build_model(self, config_path, checkpoint_path):
        """Build BEVFusion model"""
        from mmengine.config import Config
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        
        register_all_modules()
        
        config = Config.fromfile(config_path)
        model = MODELS.build(config.model)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        
        return model.to(self.device)
    
    def validate_and_fix_data(self, data_batch):
        """Validate and fix data format (YOUR WORKING METHOD)"""
        print(f"DEBUG: validate_and_fix_data called with: {type(data_batch)}")
        if isinstance(data_batch, dict):
            print(f"DEBUG: data_batch keys: {data_batch.keys()}")
        
        try:
            if not isinstance(data_batch, dict) or 'inputs' not in data_batch:
                print(f"DEBUG: FAILED - not dict or no 'inputs' key")
                return None
            
            batch_inputs_dict = data_batch['inputs']
            print(f"DEBUG: batch_inputs_dict type: {type(batch_inputs_dict)}")
            print(f"DEBUG: batch_inputs_dict keys: {batch_inputs_dict.keys()}")
            
            # Fix points format - CRITICAL for BEVFusion
            if 'points' in batch_inputs_dict:
                points = batch_inputs_dict['points']
                print(f"DEBUG: Original points type: {type(points)}")
                if isinstance(points, torch.Tensor):
                    print(f"DEBUG: Points tensor shape: {points.shape}")
                    # Validate points tensor
                    if points.numel() == 0:
                        print(f"DEBUG: FAILED - Empty points tensor")
                        return None
                    
                    if len(points.shape) != 2 or points.shape[1] < 3:
                        print(f"DEBUG: FAILED - Invalid points shape: {points.shape}, expected [N, >=3]")
                        return None
                    
                    # CRITICAL: Additional point cloud validation for sparse convolution
                    if torch.isnan(points).any() or torch.isinf(points).any():
                        print(f"DEBUG: FAILED - Points contain NaN or Inf values")
                        return None
                    
                    # Check if points are within reasonable range (prevent sparse conv issues)
                    xyz = points[:, :3]
                    if (xyz.abs() > 1000).any():
                        print(f"DEBUG: WARNING - Points have very large coordinates, clipping...")
                        xyz = torch.clamp(xyz, -1000, 1000)
                        points[:, :3] = xyz
                    
                    # Ensure minimum points for processing
                    if points.shape[0] < 100:
                        print(f"DEBUG: FAILED - Too few points ({points.shape[0]}), need at least 100")
                        return None
                    
                    # CRITICAL: Ensure points are on the correct device
                    points = points.to(self.device)
                    batch_inputs_dict['points'] = [points]  # BEVFusion expects list format
                    print(f"DEBUG: SUCCESS - Fixed points format: {points.shape} -> list[{points.shape}]")
                elif isinstance(points, list):
                    print(f"DEBUG: Points already list with {len(points)} items")
                    for i, p in enumerate(points):
                        if isinstance(p, torch.Tensor):
                            # Apply same validation to list items
                            if torch.isnan(p).any() or torch.isinf(p).any():
                                print(f"DEBUG: FAILED - Points[{i}] contain NaN or Inf values")
                                return None
                            if p.shape[0] < 100:
                                print(f"DEBUG: FAILED - Points[{i}] too few points ({p.shape[0]})")
                                return None
                            points[i] = p.to(self.device)
                            print(f"DEBUG: Moved points[{i}] to device: {p.shape}")
                else:
                    print(f"DEBUG: FAILED - Unknown points type: {type(points)}")
                    return None
            else:
                print(f"DEBUG: FAILED - No 'points' key in batch_inputs_dict")
                return None
            
            # Fix data_samples device issues
            batch_data_samples = data_batch.get('data_samples', [])
            print(f"DEBUG: data_samples type: {type(batch_data_samples)}")
            if batch_data_samples:
                # Handle both single sample and list of samples
                if not isinstance(batch_data_samples, list):
                    batch_data_samples = [batch_data_samples]
                    print(f"DEBUG: Converted data_samples to list")
                
                print(f"DEBUG: Processing {len(batch_data_samples)} data samples")
                # Move all tensors to device
                for i, sample in enumerate(batch_data_samples):
                    if hasattr(sample, 'gt_instances_3d') and sample.gt_instances_3d is not None:
                        gt_instances = sample.gt_instances_3d
                        # Move tensor attributes to device
                        for attr_name in dir(gt_instances):
                            if not attr_name.startswith('_') and hasattr(gt_instances, attr_name):
                                try:
                                    attr_value = getattr(gt_instances, attr_name)
                                    if isinstance(attr_value, torch.Tensor):
                                        setattr(gt_instances, attr_name, attr_value.to(self.device))
                                except:
                                    pass
                
                data_batch['data_samples'] = batch_data_samples
            
            # Ensure other inputs are on correct device
            for key, value in batch_inputs_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_inputs_dict[key] = value.to(self.device)
                elif isinstance(value, list):
                    for j, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            value[j] = item.to(self.device)
            
            print(f"DEBUG: SUCCESS - Data validation passed")
            return data_batch
            
        except Exception as e:
            print(f"DEBUG: EXCEPTION in validate_and_fix_data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_features(self, sample):
        """Extract features from single sample (TRUE MULTI-LEVEL)"""
        try:
            # Clear GPU cache before processing to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
            
            # Validate and fix data format
            data_batch = self.validate_and_fix_data(sample)
            if data_batch is None:
                print(f"DEBUG: validate_and_fix_data returned None")
                return None
            
            batch_inputs_dict = data_batch['inputs']
            batch_data_samples = data_batch.get('data_samples', [])
            
            # Prepare batch_input_metas for BEVFusion (extracted from data_samples)
            batch_input_metas = None
            if batch_data_samples:
                try:
                    batch_input_metas = [item.metainfo for item in batch_data_samples if hasattr(item, 'metainfo')]
                    print(f"DEBUG: Extracted {len(batch_input_metas)} metainfo items")
                except:
                    batch_input_metas = None
                    print(f"DEBUG: Could not extract metainfo, using None")
            
            # 🔥 TRUE MULTI-LEVEL EXTRACTION: Hook into ALL layers
            teacher_features_dict = {}
            student_features_dict = {}
            
            # Setup hooks for multi-level extraction
            teacher_hooks = []
            student_hooks = []
            
            def create_hook(features_dict, level_name):
                def hook_fn(module, input, output):
                    try:
                        if isinstance(output, (list, tuple)):
                            if len(output) == 1:
                                features = output[0]
                            else:
                                # FIXED: Handle backbone multi-scale outputs properly
                                if 'backbone' in level_name:
                                    # For backbone: Combine all scales using adaptive pooling
                                    scales = [f.shape for f in output]
                                    print(f"DEBUG: Backbone multi-scale: {scales}")
                                    
                                    # Resize all feature maps to the largest spatial size, then concatenate channels
                                    target_h, target_w = output[0].shape[2], output[0].shape[3]  # Use first scale as target
                                    processed_features = []
                                    
                                    for i, feat in enumerate(output):
                                        if feat.shape[2:] != (target_h, target_w):
                                            # Upsample smaller feature maps to target size
                                            feat_resized = torch.nn.functional.interpolate(
                                                feat, size=(target_h, target_w), mode='bilinear', align_corners=False
                                            )
                                            processed_features.append(feat_resized)
                                            print(f"DEBUG: Resized scale {i}: {feat.shape} → {feat_resized.shape}")
                                        else:
                                            processed_features.append(feat)
                                    
                                    # Concatenate along channel dimension
                                    features = torch.cat(processed_features, dim=1)
                                    print(f"DEBUG: Backbone combined: {features.shape} from {len(output)} scales")
                                else:
                                    # For other layers: try to concatenate along channel dimension
                                    try:
                                        features = torch.cat(output, dim=1)
                                    except RuntimeError:
                                        # If concatenation fails, take the first output
                                        features = output[0]
                                        print(f"DEBUG: Concatenation failed, using first output: {features.shape}")
                        else:
                            features = output
                        features_dict[level_name] = features.detach() if level_name.startswith('teacher') else features
                        print(f"DEBUG: Captured {level_name}: {features.shape}")
                    except Exception as e:
                        print(f"DEBUG: Hook failed for {level_name}: {e}")
                        features_dict[level_name] = None
                return hook_fn
            
            # Register hooks for TEACHER (all layers: voxel + middle + backbone + neck)
            # FIXED: Voxel encoder hook - handle missing call by hooking pts_voxel_layer instead
            if hasattr(self.teacher_model, 'pts_voxel_encoder'):
                try:
                    hook = self.teacher_model.pts_voxel_encoder.register_forward_hook(
                        create_hook(teacher_features_dict, 'teacher_voxel_encoder'))
                    teacher_hooks.append(hook)
                except:
                    print("DEBUG: pts_voxel_encoder hook failed, voxel encoder may not be called in forward pass")
            
            # ALTERNATIVE: Hook the voxel layer directly (this gets called!)
            if hasattr(self.teacher_model, 'pts_voxel_layer'):
                def voxel_layer_hook(module, input, output):
                    # Process voxel layer output and simulate voxel encoder features
                    try:
                        if isinstance(output, (list, tuple)) and len(output) >= 2:
                            voxel_features = output[0]  # [N, C] voxel features
                            # Create synthetic voxel encoder features for distillation
                            teacher_features_dict['teacher_voxel_encoder'] = voxel_features.detach()
                            print(f"DEBUG: Captured teacher_voxel_layer (synthetic): {voxel_features.shape}")
                    except Exception as e:
                        print(f"DEBUG: Voxel layer hook failed: {e}")
                
                hook = self.teacher_model.pts_voxel_layer.register_forward_hook(voxel_layer_hook)
                teacher_hooks.append(hook)
                
            if hasattr(self.teacher_model, 'pts_middle_encoder'):
                hook = self.teacher_model.pts_middle_encoder.register_forward_hook(
                    create_hook(teacher_features_dict, 'teacher_middle_encoder'))
                teacher_hooks.append(hook)
                
            if hasattr(self.teacher_model, 'pts_backbone'):
                hook = self.teacher_model.pts_backbone.register_forward_hook(
                    create_hook(teacher_features_dict, 'teacher_backbone'))
                teacher_hooks.append(hook)
                
            if hasattr(self.teacher_model, 'pts_neck'):
                hook = self.teacher_model.pts_neck.register_forward_hook(
                    create_hook(teacher_features_dict, 'teacher_neck'))
                teacher_hooks.append(hook)
            
            # Register hooks for STUDENT (all layers: voxel + middle + backbone + neck)
            # FIXED: Voxel encoder hook - handle missing call by hooking pts_voxel_layer instead
            if hasattr(self.student_model, 'pts_voxel_encoder'):
                try:
                    hook = self.student_model.pts_voxel_encoder.register_forward_hook(
                        create_hook(student_features_dict, 'student_voxel_encoder'))
                    student_hooks.append(hook)
                except:
                    print("DEBUG: pts_voxel_encoder hook failed, voxel encoder may not be called in forward pass")
            
            # ALTERNATIVE: Hook the voxel layer directly (this gets called!)
            if hasattr(self.student_model, 'pts_voxel_layer'):
                def voxel_layer_hook(module, input, output):
                    # Process voxel layer output and simulate voxel encoder features
                    try:
                        if isinstance(output, (list, tuple)) and len(output) >= 2:
                            voxel_features = output[0]  # [N, C] voxel features
                            # Create synthetic voxel encoder features for distillation (requires grad!)
                            student_features_dict['student_voxel_encoder'] = voxel_features  # No detach for student!
                            print(f"DEBUG: Captured student_voxel_layer (synthetic): {voxel_features.shape}")
                    except Exception as e:
                        print(f"DEBUG: Voxel layer hook failed: {e}")
                
                hook = self.student_model.pts_voxel_layer.register_forward_hook(voxel_layer_hook)
                student_hooks.append(hook)
                
            if hasattr(self.student_model, 'pts_middle_encoder'):
                hook = self.student_model.pts_middle_encoder.register_forward_hook(
                    create_hook(student_features_dict, 'student_middle_encoder'))
                student_hooks.append(hook)
                
            if hasattr(self.student_model, 'pts_backbone'):
                hook = self.student_model.pts_backbone.register_forward_hook(
                    create_hook(student_features_dict, 'student_backbone'))
                student_hooks.append(hook)
                
            if hasattr(self.student_model, 'pts_neck'):
                hook = self.student_model.pts_neck.register_forward_hook(
                    create_hook(student_features_dict, 'student_neck'))
                student_hooks.append(hook)
            
            try:
                # 1. TEACHER: DIRECT EXTRACTION with hooks capturing intermediate layers
                with torch.no_grad():
                    try:
                        self.teacher_model.eval()
                        print(f"DEBUG: Teacher MULTI-LEVEL extraction (voxel_encoder + middle_encoder + backbone + neck)...")
                        raw_teacher_features = self.teacher_model.extract_feat(batch_inputs_dict, None)
                        
                        print(f"DEBUG: Teacher extracted {len(teacher_features_dict)} feature levels")
                        for level_name, features in teacher_features_dict.items():
                            if features is not None:
                                print(f"DEBUG: Teacher {level_name}: {features.shape}")
                        
                        if len(teacher_features_dict) == 0:
                            print(f"DEBUG: No teacher features extracted!")
                            return None
                        
                    except Exception as e:
                        print(f"DEBUG: Teacher MULTI-LEVEL extraction failed: {e}")
                        return None
                
                # 2. STUDENT: DIRECT EXTRACTION with hooks capturing intermediate layers  
                try:
                    self.student_model.train()
                    print(f"DEBUG: Student MULTI-LEVEL extraction (voxel_encoder + middle_encoder + backbone + neck)...")
                    raw_student_features = self.student_model.extract_feat(batch_inputs_dict, None)
                    
                    print(f"DEBUG: Student extracted {len(student_features_dict)} feature levels")
                    for level_name, features in student_features_dict.items():
                        if features is not None:
                            print(f"DEBUG: Student {level_name}: {features.shape}")
                    
                    if len(student_features_dict) == 0:
                        print(f"DEBUG: No student features extracted!")
                        return None
                        
                except Exception as e:
                    print(f"DEBUG: Student MULTI-LEVEL extraction failed: {e}")
                    return None
                
            finally:
                # Cleanup hooks
                for hook in teacher_hooks:
                    hook.remove()
                for hook in student_hooks:
                    hook.remove()
            
            # Create level mappings (voxel_encoder -> middle_encoder -> backbone -> neck)
            multi_level_features = {}
            
            # Map teacher and student features to common level names (ALL 4 LEVELS!)
            level_mappings = [
                ('voxel_encoder', 'teacher_voxel_encoder', 'student_voxel_encoder'),
                ('middle_encoder', 'teacher_middle_encoder', 'student_middle_encoder'),
                ('backbone', 'teacher_backbone', 'student_backbone'), 
                ('neck', 'teacher_neck', 'student_neck')
            ]
            
            print(f"DEBUG: Checking for 4-level distillation (voxel_encoder + middle_encoder + backbone + neck)...")
            
            for level_name, teacher_key, student_key in level_mappings:
                if teacher_key in teacher_features_dict and student_key in student_features_dict:
                    if teacher_features_dict[teacher_key] is not None and student_features_dict[student_key] is not None:
                        multi_level_features[level_name] = {
                            'teacher': teacher_features_dict[teacher_key],
                            'student': student_features_dict[student_key]
                        }
                        print(f"DEBUG: ✅ Level '{level_name}' - AVAILABLE for distillation")
                    else:
                        print(f"DEBUG: ❌ Level '{level_name}' - FAILED (teacher={teacher_features_dict[teacher_key] is not None}, student={student_features_dict[student_key] is not None})")
                else:
                    print(f"DEBUG: ❌ Level '{level_name}' - MISSING (teacher_key='{teacher_key}' in dict: {teacher_key in teacher_features_dict}, student_key='{student_key}' in dict: {student_key in student_features_dict})")
            
            print(f"DEBUG: Multi-level features available: {list(multi_level_features.keys())}")
            
            return {
                'multi_level_features': multi_level_features,
                'teacher_features_dict': teacher_features_dict,
                'student_features_dict': student_features_dict
            }
            
        except Exception as e:
            print(f"  Feature extraction failed: {e}")
            return None

    def train_batch(self, data_batch):
        """Train on batch using individual sample processing with MULTI-LEVEL features"""
        print(f"DEBUG: Processing batch of {len(data_batch)} samples individually (MULTI-LEVEL)")
        print(f"DEBUG: data_batch type: {type(data_batch)}")
        print(f"DEBUG: First sample type: {type(data_batch[0]) if len(data_batch) > 0 else 'empty'}")
        
        teacher_features_batch = []
        student_features_batch = []
        multi_level_features_batch = []  # NEW: Collect paired multi-level features
        successful_samples = 0
        
        # Extract features from each sample individually with robust error handling
        for i, sample in enumerate(data_batch):
            print(f"DEBUG: Processing sample {i}, type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"DEBUG: Sample {i} keys: {sample.keys()}")
            try:
                features = self.extract_features(sample)
                if (features and features['multi_level_features']):
                    
                    # FIXED: Collect paired multi-level features for batch processing
                    multi_level_features_batch.append(features['multi_level_features'])
                    successful_samples += 1
                    
                    # Showing true multi-level extraction results
                    available_levels = list(features['multi_level_features'].keys())
                    print(f"DEBUG: Sample {i}: Got TRUE multi-level features: {available_levels}")
                    for level_name in available_levels:
                        level_data = features['multi_level_features'][level_name]
                        teacher_shape = level_data['teacher'].shape
                        student_shape = level_data['student'].shape
                        print(f"  Level '{level_name}': Teacher {teacher_shape}, Student {student_shape}")
                else:
                    print(f"DEBUG: Sample {i}: Failed to extract features (validation failed)")
            except Exception as e:
                print(f"DEBUG: Sample {i}: Exception during feature extraction: {str(e)[:100]}...")
                continue
        
        if successful_samples < 2:
            print("DEBUG: Need at least 2 samples for InfoNCE")
            return 0.0
        
        print(f"DEBUG: Success rate: {successful_samples}/{len(data_batch)}")
        
        # Use multi-level compute_batch_clip_loss approach
        try:
            self.optimizer.zero_grad()
            
            infonce_loss = self.compute_batch_clip_loss_multilevel_paired(
                multi_level_features_batch
            )
            
            if infonce_loss.item() > 0:
                infonce_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                print(f"DEBUG: Multi-level InfoNCE loss: {infonce_loss.item():.4f} (SUCCESS: > 0)")
                return infonce_loss.item()
            else:
                print(f"DEBUG: Multi-level InfoNCE loss still 0, skipping backward pass")
                return 0.0
            
        except Exception as e:
            print(f"DEBUG: Multi-level batch processing failed: {e}")
            return 0.0
    
    def compute_batch_clip_loss_multilevel_paired(self, multi_level_features_batch):
        """Compute InfoNCE loss on batch of PAIRED MULTI-LEVEL features"""
        try:
            # Combine multi-level features from all samples 
            combined_teacher_dict = {}
            combined_student_dict = {}
            
            # Get all available levels from paired data
            all_levels = set()
            for sample_features in multi_level_features_batch:
                all_levels.update(sample_features.keys())
            
            print(f"DEBUG: Available levels for paired multi-level distillation: {list(all_levels)}")
            
            # Concatenate features for each level
            for level_name in all_levels:
                teacher_level_features = []
                student_level_features = []
                
                for i, sample_features in enumerate(multi_level_features_batch):
                    if level_name in sample_features:
                        level_data = sample_features[level_name]
                        if 'teacher' in level_data and 'student' in level_data:
                            if level_data['teacher'] is not None and level_data['student'] is not None:
                                teacher_level_features.append(level_data['teacher'])
                                student_level_features.append(level_data['student'])
                
                if len(teacher_level_features) >= 2:  # Need at least 2 for InfoNCE
                    combined_teacher_dict[level_name] = torch.cat(teacher_level_features, dim=0)
                    combined_student_dict[level_name] = torch.cat(student_level_features, dim=0)
                    print(f"DEBUG: Level '{level_name}' - Teacher: {combined_teacher_dict[level_name].shape}, Student: {combined_student_dict[level_name].shape}")
                else:
                    print(f"DEBUG: Level '{level_name}' - Insufficient samples ({len(teacher_level_features)}), skipping")
            
            if not combined_teacher_dict or not combined_student_dict:
                print("DEBUG: No valid levels for paired multi-level distillation")
                return torch.tensor(0.0, device=self.device)
            
            # Get CLIP model output with COMBINED-ONLY InfoNCE
            loss_dict = self.clip_model(combined_teacher_dict, combined_student_dict)
            
            # Add student projection params to optimizer after creation
            self._add_student_projection_to_optimizer()
            
            # Extract the main loss from the dictionary
            if 'clip_total_loss' in loss_dict:
                infonce_loss = loss_dict['clip_total_loss']
                print(f"DEBUG: COMBINED-ONLY InfoNCE LOSS: {infonce_loss.item():.4f}")
                
                # Show additional info from combined-only approach
                if 'processed_levels' in loss_dict:
                    levels = loss_dict['processed_levels']
                    print(f"DEBUG: Processed levels: {levels}")
                if 'embedding_dim' in loss_dict:
                    dim = loss_dict['embedding_dim']
                    print(f"DEBUG: Combined embedding dimension: {dim}")
                if 'approach' in loss_dict:
                    approach = loss_dict['approach']
                    print(f"DEBUG: Distillation approach: {approach}")
                
                return infonce_loss
            else:
                print("DEBUG: CLIP model returned invalid loss dict")
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            print(f"DEBUG: Paired multi-level loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)

    # REMOVED: Old compute_batch_clip_loss_multilevel function (dead code)
    # Now using compute_batch_clip_loss_multilevel_paired with combined-only InfoNCE
    
    def _add_student_projection_to_optimizer(self):
        """Add ONLY student projection parameters to optimizer when created (MULTI-LEVEL)"""
        # Check all student projections (for multi-level distillation)
        current_params = set()
        for group in self.optimizer.param_groups:
            current_params.update(id(p) for p in group['params'])
        
        new_params = []
        
        # Find all student→teacher projection attributes
        for attr_name in dir(self.clip_model):
            if attr_name.startswith('student_') and 'to_teacher' in attr_name:
                projection = getattr(self.clip_model, attr_name, None)
                if projection is not None and hasattr(projection, 'parameters'):
                    for param in projection.parameters():
                        if id(param) not in current_params and param.requires_grad:
                            new_params.append(param)
        
        # UPDATED: No teacher projections in new approach (teacher features used directly)
        print(f"✅ CORRECT DISTILLATION: No teacher projections (using native teacher features directly)")
        
        if new_params:
            self.optimizer.add_param_group({'params': new_params})
            print(f"DEBUG: Added {len(new_params)} STUDENT-ONLY projection parameters to optimizer (multi-level)")
        else:
            print("DEBUG: No new student projection parameters to add")

    def train(self, dataloader, num_epochs=20, save_dir="checkpoints"):
        """Training loop with checkpoint saving"""
        print(f"Training for {num_epochs} epochs...")
        
        # Create checkpoint directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.student_model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, data_batch in enumerate(dataloader):
                # Process batch for InfoNCE with validation
                if isinstance(data_batch, list) and len(data_batch) >= 2:
                    loss = self.train_batch(data_batch)
                else:
                    print(f"Batch {batch_idx}: Only {len(data_batch) if isinstance(data_batch, list) else 1} sample(s), skipping InfoNCE")
                    loss = 0.0
                
                if loss > 0:
                    total_loss += loss
                    num_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1} done - Average Loss: {avg_loss:.4f}")
                
                # Save checkpoint after each epoch
                self.save_checkpoint(epoch + 1, avg_loss, save_dir)
            else:
                print(f"Epoch {epoch+1} - No valid batches")

    def save_checkpoint(self, epoch, avg_loss, save_dir):
        """Save checkpoint in official BEVFusion format compatible with tools/test.py"""
        
        # Save in official BEVFusion format
        checkpoint = {
            'state_dict': self.student_model.state_dict(),  # ✅ Official BEVFusion format
            'epoch': epoch,
            'meta': {
                'distillation_loss': avg_loss,
                'approach': 'combined_multilevel_infonce_distillation'
            }
        }
        
        # Save checkpoint with epoch info
        checkpoint_path = f"{save_dir}/epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest
        latest_path = f"{save_dir}/latest.pth"
        torch.save(checkpoint, latest_path)
        print(f"✅ Latest checkpoint saved: {latest_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint in official BEVFusion format"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load official BEVFusion format
        if 'state_dict' in checkpoint:
            self.student_model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint.get('epoch', 0)
            avg_loss = checkpoint.get('meta', {}).get('distillation_loss', 0.0)
            print(f"✅ Checkpoint loaded: Epoch {epoch}, Loss {avg_loss:.4f}")
        else:
            # Direct state_dict (fallback)
            self.student_model.load_state_dict(checkpoint)
            epoch = 0
            avg_loss = 0.0
            print(f"✅ Direct state_dict loaded")
            
        return epoch, avg_loss


def main():
    """Simple main function"""
    print("Simple BEVFusion Distillation")
    
    # Paths
    teacher_config = 'work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
    teacher_checkpoint = 'work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth'
    student_config = 'configs/bevfusion_lidar_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d_lightweight_singapore.py'
    
    # Create trainer
    trainer = SimpleTrainer(teacher_config, teacher_checkpoint, student_config)
    
    # Build simple dataset
    from mmengine.config import Config
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules
    
    register_all_modules()
    
    cfg = Config.fromfile(student_config)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # FIXED dataloader - Return ALL samples for InfoNCE
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Get 4 samples for InfoNCE contrastive learning
        shuffle=True,
        collate_fn=lambda x: x,  # Return ALL samples as list (not just first one!)
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Optional: Resume from checkpoint
    # resume_checkpoint = "work_dirs/distillation_checkpoints/latest.pth"
    # if os.path.exists(resume_checkpoint):
    #     epoch, loss = trainer.load_checkpoint(resume_checkpoint)
    #     print(f"Resumed training from epoch {epoch}")
    
    # Train with checkpoint saving
    trainer.train(dataloader, num_epochs=5, save_dir="work_dirs/distillation_checkpoints")
    
    print("Done!")


if __name__ == '__main__':
    main() 