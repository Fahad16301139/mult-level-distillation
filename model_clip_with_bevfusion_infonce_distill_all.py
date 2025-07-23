import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class CLIP(nn.Module):
    """CLIP model for BEVFusion distillation with InfoNCE - MATCHING WORKING SCRIPT INTERFACE"""
    
    def __init__(self, embed_dim: int = 512, teacher_channels: int = 512, student_channels: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        
        # Learnable temperature parameter (like CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Projection layers for combined-only approach (created dynamically)
        self.student_projections = nn.ModuleDict()
        
        # Track if projections are initialized
        self._projections_init = False
        
        print(f"DEBUG: CLIP model created - embed_dim={embed_dim}, teacher_ch={teacher_channels}, student_ch={student_channels}")
    
    def _pool_features(self, features: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """
        Pool features to consistent size for concatenation.
        
        Args:
            features: Input tensor [B, C, H, W], [B, C, D, H, W], or [N_voxels, features] for voxel encoder
            batch_size: Expected batch size (needed for voxel encoder reshaping)
            
        Returns:
            Pooled tensor [B, C] 
        """
        # Handle voxel encoder special case: [N_voxels, feat1, feat2] -> [B, C]
        if features.dim() == 3 and batch_size is not None and features.shape[0] > batch_size * 10:
            # Voxel encoder case: [N_voxels, feat1, feat2] -> [B, combined_features]
            print(f"DEBUG: Detected voxel encoder - converting {features.shape} to batch format")
            
            n_voxels, feat1, feat2 = features.shape
            voxels_per_batch = n_voxels // batch_size
            
            # Flatten last two dimensions first: [N_voxels, feat1, feat2] -> [N_voxels, feat1*feat2]
            features_flat = features.flatten(1)  # [N_voxels, feat1*feat2]
            
            # Reshape to batch format and pool
            try:
                # Take even number of voxels per batch
                usable_voxels = batch_size * voxels_per_batch
                features_batched = features_flat[:usable_voxels].view(batch_size, voxels_per_batch, -1)
                # Pool over voxels: [B, voxels_per_batch, features] -> [B, features]
                pooled = features_batched.mean(dim=1)  # Average pool over voxels
                print(f"DEBUG: Voxel conversion successful: {features.shape} -> {pooled.shape}")
                return pooled
            except Exception as e:
                print(f"DEBUG: Voxel reshaping failed: {e}, using global pooling")
                # Fallback: global pooling across all voxels
                features_flat = features.flatten(1)  # [N_voxels, feat1*feat2]
                global_avg = features_flat.mean(dim=0)  # [feat1*feat2]
                return global_avg.unsqueeze(0).repeat(batch_size, 1)  # [B, feat1*feat2]
        
        # Standard cases for other levels
        if features.dim() == 4:  # [B, C, H, W]
            return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.dim() == 5:  # [B, C, D, H, W] 
            return F.adaptive_avg_pool3d(features, (1, 1, 1)).flatten(1)
        elif features.dim() == 3:  # [B, C, L]
            return F.adaptive_avg_pool1d(features, 1).flatten(1)
        elif features.dim() == 2:  # [B, C] - already pooled
            return features
        else:
            raise ValueError(f"Unsupported tensor shape: {features.shape}")
    
    def _create_projection(self, input_dim: int, output_dim: int, device: torch.device) -> nn.Module:
        """Create a simple linear projection layer"""
        return nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, teacher_features, student_features) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing COMBINED-ONLY InfoNCE:
        - Extract and pool features from all levels
        - Concatenate into single combined embedding  
        - Single InfoNCE on complete pipeline representation
        """
        device = next(iter(teacher_features.values())).device
        
        # Collect embeddings from all levels
        teacher_embeddings = []
        student_embeddings = []
        processed_levels = []
        
        print(f"DEBUG: Processing {len(teacher_features)} teacher levels, {len(student_features)} student levels")
        
        # Detect batch size from regular feature maps (not voxel encoder)
        batch_size = None
        for level_name, features in teacher_features.items():
            if features is not None and features.dim() >= 4:  # Regular feature maps [B, C, ...]
                batch_size = features.shape[0]
                break
        
        print(f"DEBUG: Detected batch size: {batch_size}")
        
        # Extract and pool features from each level
        for level_name in teacher_features.keys():
            if level_name in student_features:
                t_feat = teacher_features[level_name]
                s_feat = student_features[level_name]
                
                if t_feat is not None and s_feat is not None:
                    print(f"DEBUG: Processing level '{level_name}' - Teacher: {t_feat.shape}, Student: {s_feat.shape}")
                    
                    try:
                        # Pool to consistent embedding size (pass batch_size for voxel encoder)
                        teacher_pooled = self._pool_features(t_feat, batch_size)  # [B, C]
                        student_pooled = self._pool_features(s_feat, batch_size)  # [B, C]
                        
                        teacher_embeddings.append(teacher_pooled)
                        student_embeddings.append(student_pooled)
                        processed_levels.append(level_name)
                        
                        print(f"DEBUG: Level '{level_name}' pooled - Teacher: {teacher_pooled.shape}, Student: {student_pooled.shape}")
                        
                    except Exception as e:
                        print(f"ERROR: Failed to process level '{level_name}': {e}")
                        continue
        
        # Create combined embeddings
        if len(teacher_embeddings) >= 2:
            try:
                print(f"DEBUG: Creating combined embedding from {len(teacher_embeddings)} levels: {processed_levels}")
                
                # Concatenate ALL level embeddings
                teacher_combined = torch.cat(teacher_embeddings, dim=1)  # [B, total_channels]
                student_combined = torch.cat(student_embeddings, dim=1)  # [B, total_channels]
                
                print(f"DEBUG: Combined embeddings - Teacher: {teacher_combined.shape}, Student: {student_combined.shape}")
                
                # NO PROJECTION - Use adaptive pooling to match dimensions
                teacher_dim = teacher_combined.shape[1]  # 1202
                student_dim = student_combined.shape[1]  # 754
                
                print(f"DEBUG: Adaptive pooling: teacher_dim={teacher_dim}, student_dim={student_dim}")
                
                # Pool both to the same target dimension (use minimum for efficiency)
                target_dim = min(teacher_dim, student_dim)  # 754
                
                if teacher_dim > target_dim:
                    # Pool teacher down: [B, 1202] -> [B, 754]
                    teacher_final = F.adaptive_avg_pool1d(
                        teacher_combined.unsqueeze(1), target_dim
                    ).squeeze(1)
                else:
                    teacher_final = teacher_combined
                    
                if student_dim > target_dim:
                    # Pool student down: [B, student_dim] -> [B, 754] 
                    student_final = F.adaptive_avg_pool1d(
                        student_combined.unsqueeze(1), target_dim
                    ).squeeze(1)
                else:
                    student_final = student_combined
                
                print(f"DEBUG: After pooling - Teacher: {teacher_final.shape}, Student: {student_final.shape}")
                
                # Normalize embeddings
                teacher_norm = F.normalize(teacher_final, dim=1)
                student_norm = F.normalize(student_final, dim=1)
                
                # Compute cosine similarity
                temperature = 0.07
                logits = torch.mm(student_norm, teacher_norm.T) / temperature  # [B, B]
                
                # InfoNCE loss (one-directional: student learns from teacher)
                targets = torch.arange(teacher_norm.size(0)).to(device)
                combined_loss = F.cross_entropy(logits, targets)
                
                print(f"DEBUG: COMBINED-ONLY InfoNCE loss: {combined_loss.item():.4f}")
                print(f"DEBUG: Processed levels: {processed_levels}")
                print(f"DEBUG: Final embedding dimension: {target_dim}")
                
                # Return single combined loss
                result = {
                    'clip_total_loss': combined_loss,
                    'num_levels': len(processed_levels),
                    'processed_levels': processed_levels,
                    'embedding_dim': target_dim,
                    'approach': 'combined_only'
                }
                return result
                
            except Exception as e:
                print(f"ERROR: Failed to create combined embedding: {e}")
                return {'clip_total_loss': torch.tensor(0.0, device=device, requires_grad=True)}
        
        else:
            print(f"WARNING: Not enough levels processed! Only {len(teacher_embeddings)} levels available.")
            return {'clip_total_loss': torch.tensor(0.0, device=device, requires_grad=True)}
    
    def _process_single_level(self, teacher_features: torch.Tensor, student_features: torch.Tensor, 
                             level_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single level of features"""
        
        # Pool features to 2D format [batch_size, channels]
        teacher_pooled = self._pool_features(teacher_features)  # [B, teacher_channels]
        student_pooled = self._pool_features(student_features)  # [B, student_channels]
        
        batch_size = teacher_pooled.shape[0]
        
        # DETECT ACTUAL DIMENSIONS AT RUNTIME (FIXED)
        actual_teacher_channels = teacher_pooled.shape[1]
        actual_student_channels = student_pooled.shape[1]
        
        print(f"DEBUG: Level '{level_name}' - Teacher: {actual_teacher_channels}, Student: {actual_student_channels}")
        
        # CORRECT DISTILLATION: Only student projects to teacher's native space
        student_proj_name = f'student_{level_name}_to_teacher'
        
        # Create student projection to teacher's native feature space (trainable)
        if not hasattr(self, student_proj_name) or getattr(self, student_proj_name) is None:
            student_projection = self._create_projection(actual_student_channels, actual_teacher_channels, device)
            setattr(self, student_proj_name, student_projection)
            self._student_init = True
            print(f"DEBUG: Created TRAINABLE student→teacher projection for '{level_name}': {actual_student_channels} → {actual_teacher_channels}")
        
        # Get student projection
        student_projection = getattr(self, student_proj_name)
        
        # CORRECT APPROACH: Teacher features used directly, student projected to teacher space
        with torch.no_grad():
            teacher_embed = teacher_pooled                     # [B, teacher_channels] - NATIVE SPACE
        student_embed = student_projection(student_pooled)     # [B, teacher_channels] - PROJECTED TO TEACHER SPACE
        
        print(f"DEBUG: Teacher native: {teacher_embed.shape}, Student projected: {student_embed.shape}")
        
        # L2 normalize embeddings (CRITICAL for cosine similarity)
        teacher_norm = F.normalize(teacher_embed, dim=1)  # [B, embed_dim]
        student_norm = F.normalize(student_embed, dim=1)  # [B, embed_dim]
        
        # Compute cosine similarity matrix
        raw_cosine_sim = teacher_norm @ student_norm.T  # [B, B] with values in [-1, 1]
        
        # Apply temperature scaling for logits (like CLIP)
        logit_scale = self.logit_scale.exp()
        logits_teacher = logit_scale * raw_cosine_sim      # [B, B] scaled similarities
        logits_student = logits_teacher.T                 # [B, B] transposed
        
        return logits_teacher, logits_student, raw_cosine_sim


class BEVFusionCLIPWrapper(nn.Module):
    """Extract features from BEVFusion - SIMPLIFIED"""
    
    def __init__(self, bevfusion_model, is_teacher: bool = True):
        super().__init__()
        self.bevfusion_model = bevfusion_model
        self.is_teacher = is_teacher
        self.features = {}
        
        if is_teacher:
            # Freeze teacher
            for param in self.bevfusion_model.parameters():
                param.requires_grad = False
            self.bevfusion_model.eval()
        
        # Register hooks to capture features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture features"""
        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(output, (list, tuple)):
                    self.features[name] = output[0] if len(output) == 1 else torch.cat(output, dim=1)
                else:
                    self.features[name] = output
            return fn
        
        # Hook different parts of the model
        if hasattr(self.bevfusion_model, 'pts_voxel_encoder'):
            self.bevfusion_model.pts_voxel_encoder.register_forward_hook(hook_fn('voxel_encoder'))
        if hasattr(self.bevfusion_model, 'pts_middle_encoder'):
            self.bevfusion_model.pts_middle_encoder.register_forward_hook(hook_fn('middle_encoder'))
        if hasattr(self.bevfusion_model, 'pts_backbone'):
            self.bevfusion_model.pts_backbone.register_forward_hook(hook_fn('backbone'))
        if hasattr(self.bevfusion_model, 'pts_neck'):
            self.bevfusion_model.pts_neck.register_forward_hook(hook_fn('neck'))
    
    def forward(self, batch_inputs: dict, batch_input_metas: list = None):
        """Extract features with AGGRESSIVE memory management"""
        self.features.clear()
        
        try:
            # Ensure batch_input_metas is provided (REQUIRED for BEVFusion)
            if batch_input_metas is None:
                batch_input_metas = []
                print("DEBUG: Warning - batch_input_metas is None, using empty list")
            
            # Ensure model is in correct mode
            if self.is_teacher:
                self.bevfusion_model.eval()
                with torch.no_grad():
                    # CRITICAL: Use extract_feat with batch_input_metas (REQUIRED!)
                    output = self.bevfusion_model.extract_feat(batch_inputs, batch_input_metas)
            else:
                self.bevfusion_model.train()
                # CRITICAL: Use extract_feat with batch_input_metas (REQUIRED!)
                output = self.bevfusion_model.extract_feat(batch_inputs, batch_input_metas)
            
            
            extracted_features = self.features
            self.features = {}  # Reset for next extraction
            
            return output, extracted_features
            
        except RuntimeError as e:
            if "cuda" in str(e).lower() and "memory" in str(e).lower():
                print(f"CUDA memory error in feature extraction: {e}")
                # Force memory cleanup on CUDA errors
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                print(f"Feature extraction failed: {e}")
            
            self.features.clear()
            return None, {}
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            self.features.clear()
            return None, {}


class CLIPBEVFusionInfoNCELoss(nn.Module):
    """InfoNCE loss for CLIP distillation - MATCHING WORKING SCRIPT INTERFACE"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits_teacher: torch.Tensor, logits_student: torch.Tensor, 
                raw_cosine_sim: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss from logits
        
        Args:
            logits_teacher: [batch_size, batch_size] scaled similarity matrix
            logits_student: [batch_size, batch_size] scaled similarity matrix (transposed)
            raw_cosine_sim: [batch_size, batch_size] raw cosine similarities (optional)
        
        Returns:
            dict with 'clip_total_loss' key
        """
        batch_size = logits_teacher.shape[0]
        
        if batch_size < 2:
            # Can't compute contrastive loss with single sample
            return {'clip_total_loss': torch.tensor(0.0, device=logits_teacher.device)}
        
        # InfoNCE loss: for each teacher_i, student_i should be most similar
        # Ground truth: diagonal should be 1, off-diagonal should be 0
        targets = torch.arange(batch_size, device=logits_teacher.device)
        
        
        # Student should learn to produce embeddings similar to teacher
        loss_student_to_teacher = F.cross_entropy(logits_student, targets)
        
        
        # loss_teacher_to_student = F.cross_entropy(logits_teacher, targets)  # REMOVED
        
        
        total_loss = loss_student_to_teacher
        
        print(f"DEBUG: ONE-DIRECTIONAL distillation loss: {total_loss.item():.4f} (student→teacher only)")
        
        return {
            'clip_total_loss': total_loss,
            'student_to_teacher_loss': loss_student_to_teacher,
            
        }


# For backward compatibility with old naming
InfoNCELoss = CLIPBEVFusionInfoNCELoss


def build_model(config_path: str, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
    """Build BEVFusion model - MATCHING WORKING SCRIPT INTERFACE"""
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
    
    return model.to(device)


def detect_dimensions_from_config(config_path: str) -> Dict[str, int]:
    """Detect feature dimensions from ALL levels - FIXED FOR 4-LEVEL DISTILLATION"""
    try:
        from mmengine.config import Config
        config = Config.fromfile(config_path)
        
        dimensions = {}
        
        # 1. Voxel encoder dimensions
        if hasattr(config.model, 'pts_voxel_encoder') and config.model.pts_voxel_encoder is not None:
            voxel_config = config.model.pts_voxel_encoder
            if hasattr(voxel_config, 'num_features'):
                dimensions['pts_voxel_encoder'] = voxel_config.num_features
            elif hasattr(voxel_config, 'out_channels'):
                dimensions['pts_voxel_encoder'] = voxel_config.out_channels
            else:
                dimensions['pts_voxel_encoder'] = 5  # Default for HardSimpleVFE
            print(f"DEBUG: Detected voxel_encoder dim: {dimensions.get('pts_voxel_encoder', 'NOT_FOUND')}")
        
        # 2. Middle encoder dimensions (from encoder_channels final layer)
        if hasattr(config.model, 'pts_middle_encoder') and config.model.pts_middle_encoder is not None:
            middle_config = config.model.pts_middle_encoder
            if hasattr(middle_config, 'encoder_channels'):
                # Extract final layer channels: ((16,16,32), (32,32,64), (64,64,128), (128,128,256))
                encoder_channels = middle_config.encoder_channels
                if isinstance(encoder_channels, (list, tuple)) and len(encoder_channels) > 0:
                    final_layer = encoder_channels[-1]  # Last layer tuple
                    if isinstance(final_layer, (list, tuple)) and len(final_layer) > 0:
                        dimensions['pts_middle_encoder'] = final_layer[-1]  # Last value in tuple
                    else:
                        dimensions['pts_middle_encoder'] = final_layer
                else:
                    dimensions['pts_middle_encoder'] = 256  # Default
            elif hasattr(middle_config, 'output_channels'):
                dimensions['pts_middle_encoder'] = middle_config.output_channels
            else:
                dimensions['pts_middle_encoder'] = 256  # Default
            print(f"DEBUG: Detected middle_encoder dim: {dimensions.get('pts_middle_encoder', 'NOT_FOUND')}")
        
        # 3. Backbone dimensions (from out_channels)
        if hasattr(config.model, 'pts_backbone') and config.model.pts_backbone is not None:
            backbone_config = config.model.pts_backbone
            if hasattr(backbone_config, 'out_channels'):
                out_channels = backbone_config.out_channels
                if isinstance(out_channels, (list, tuple)):
                    # Sum all scales for multi-scale backbone: [64, 128] → 192
                    dimensions['pts_backbone'] = sum(out_channels)
                else:
                    dimensions['pts_backbone'] = out_channels
            else:
                dimensions['pts_backbone'] = 192  # Default
            print(f"DEBUG: Detected backbone dim: {dimensions.get('pts_backbone', 'NOT_FOUND')}")
        
        # 4. Neck dimensions (from out_channels)
        if hasattr(config.model, 'pts_neck') and config.model.pts_neck is not None:
            neck_config = config.model.pts_neck
            if hasattr(neck_config, 'out_channels'):
                out_channels = neck_config.out_channels
                if isinstance(out_channels, (list, tuple)):
                    # Sum all scales for multi-scale neck: [128, 128] → 256  
                    dimensions['pts_neck'] = sum(out_channels)
                else:
                    dimensions['pts_neck'] = out_channels
            elif hasattr(neck_config, 'in_channels'):
                in_channels = neck_config.in_channels
                if isinstance(in_channels, (list, tuple)):
                    dimensions['pts_neck'] = sum(in_channels)
                else:
                    dimensions['pts_neck'] = in_channels
            else:
                dimensions['pts_neck'] = 256  # Default
            print(f"DEBUG: Detected neck dim: {dimensions.get('pts_neck', 'NOT_FOUND')}")
        
        print(f"DEBUG: Total detected dimensions: {dimensions}")
        return dimensions
        
    except Exception as e:
        print(f"ERROR: Could not detect dimensions from {config_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return default dimensions for all levels
        return {
            'pts_voxel_encoder': 5,
            'pts_middle_encoder': 256, 
            'pts_backbone': 192,
            'pts_neck': 256
        } 