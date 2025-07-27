import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from mmdet3d.registry import MODELS
from .bevfusion import BEVFusion


class BEVFusionDistillWrapper(nn.Module):
    """
    Wrapper for BEVFusion model to extract intermediate features for distillation.
    Integrates with existing BEVFusion architecture from projects/BEVFusion/bevfusion/bevfusion.py
    """
    
    def __init__(self, bevfusion_model: BEVFusion, is_student: bool = False):
        super().__init__()
        self.model = bevfusion_model
        self.is_student = is_student
        self.features = {}
        self.feature_hooks = []
        
        # Register hooks for key layers in BEVFusion
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register forward hooks to capture intermediate features from BEVFusion layers."""
        
        # Define hook targets based on BEVFusion architecture
        hook_targets = [
            ('pts_voxel_encoder', self.model.pts_voxel_encoder),
            ('pts_middle_encoder', self.model.pts_middle_encoder), 
            ('pts_backbone', self.model.pts_backbone),
            ('pts_neck', self.model.pts_neck),
        ]
        
        # Register hooks
        for name, module in hook_targets:
            hook = module.register_forward_hook(self._get_activation_hook(name))
            self.feature_hooks.append(hook)
    
    def _get_activation_hook(self, name: str):
        """Create forward hook function to store intermediate activations."""
        def hook(module, input, output):
            # Handle different output types from BEVFusion layers
            if isinstance(output, (list, tuple)):
                self.features[name] = output[0] if len(output) > 0 else output
            else:
                self.features[name] = output
        return hook
    
    def forward(self, batch_inputs_dict: Dict, batch_data_samples: List = None):
        """
        Forward pass through BEVFusion model while capturing intermediate features.
        
        Args:
            batch_inputs_dict: Dictionary containing input data (points, images, etc.)
            batch_data_samples: List of data samples with ground truth annotations
            
        Returns:
            Tuple[output, features]: Model output and captured intermediate features
        """
        # Clear previous features
        self.features = {}
        
        # Forward pass through original BEVFusion model
        if batch_data_samples is not None:
            # Training mode - compute losses
            output = self.model.loss(batch_inputs_dict, batch_data_samples)
        else:
            # Inference mode - get predictions
            output = self.model.predict(batch_inputs_dict, batch_data_samples)
            
        return output, self.features
    
    def extract_feat(self, batch_inputs_dict: Dict, batch_input_metas: List):
        """Extract features without computing losses or predictions."""
        self.features = {}
        
        # Use BEVFusion's extract_feat method
        feat_dict = self.model.extract_feat(batch_inputs_dict, batch_input_metas)
        
        return feat_dict, self.features
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks = []


def build_bevfusion_distill_model(config_dict: Dict, checkpoint_path: str = None, is_student: bool = False):
    """
    Build BEVFusion model wrapped for distillation.
    
    Args:
        config_dict: Configuration dictionary for BEVFusion model
        checkpoint_path: Path to model checkpoint (optional)
        is_student: Whether this is a student model
        
    Returns:
        BEVFusionDistillWrapper: Wrapped model ready for distillation
    """
    # Build base BEVFusion model
    bevfusion_model = MODELS.build(config_dict)
    
    # Load checkpoint if provided
    if checkpoint_path:
        from mmengine.runner import load_checkpoint
        load_checkpoint(bevfusion_model, checkpoint_path, map_location='cpu')
    
    # Wrap model for distillation
    distill_model = BEVFusionDistillWrapper(bevfusion_model, is_student=is_student)
    
    return distill_model 