import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from bevfusion_distillation import (
    BEVFusionDistillationWrapper, 
    BEVFusionDistillationLoss,
    ProjectionMLP
)


@MODELS.register_module()
class BEVFusionDistillationModel(BaseModel):
    """BEVFusion model with knowledge distillation support.
    
    This model wraps a student BEVFusion model and optionally a teacher model
    for knowledge distillation training.
    """
    
    def __init__(self,
                 student_model: dict,
                 teacher_config: Optional[dict] = None,
                 teacher_checkpoint: Optional[str] = None,
                 distillation_loss: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        
        # Don't pass data_preprocessor to parent - let student handle it
        super().__init__(init_cfg=init_cfg)
        
        # Build student model
        self.student = MODELS.build(student_model)
        
        # Use student's data preprocessor
        self.data_preprocessor = self.student.data_preprocessor
        
        # Build teacher model if provided
        self.teacher = None
        self.distillation_enabled = False
        
        if teacher_config is not None and teacher_checkpoint is not None:
            print(f"🔥 Loading teacher model from config: {teacher_config}")
            print(f"🔥 Loading teacher checkpoint: {teacher_checkpoint}")
            
            # If teacher_config is not provided, use the teacher checkpoint to infer the config
            if isinstance(teacher_config, str):
                # If it's a path to config file, load it
                from mmengine.config import Config
                teacher_cfg = Config.fromfile(teacher_config)
                self.teacher = MODELS.build(teacher_cfg.model)
            else:
                # If it's a dict config, use it directly
                self.teacher = MODELS.build(teacher_config)
            
            load_checkpoint(self.teacher, teacher_checkpoint, map_location='cpu')
            self.teacher.eval()
            
            # Move teacher to GPU and clear cache
            import torch
            if torch.cuda.is_available():
                self.teacher = self.teacher.cuda()
                torch.cuda.empty_cache()
            print("✅ Teacher model loaded and set to eval mode")
            
            # Freeze teacher parameters
            for param in self.teacher.parameters():
                param.requires_grad = False
            print("✅ Teacher parameters frozen")
            
            self.distillation_enabled = True
            
            # Wrap models for feature extraction
            self.student_wrapper = BEVFusionDistillationWrapper(self.student, is_student=True)
            self.teacher_wrapper = BEVFusionDistillationWrapper(self.teacher, is_student=False)
            print("✅ Student and teacher wrappers created")
            
            # Build distillation loss
            if distillation_loss is not None:
                self.distillation_loss = BEVFusionDistillationLoss(**distillation_loss)
            else:
                # Default distillation loss
                self.distillation_loss = BEVFusionDistillationLoss(
                    student_channels={'pts_backbone': 192, 'pts_neck': 256},
                    teacher_channels={'pts_backbone': 384, 'pts_neck': 512}
                )
            print("✅ Distillation loss initialized")
        else:
            print("⚠️ No teacher config or checkpoint provided - running student-only training")
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample]) -> dict:
        """Compute losses including distillation losses."""
        
        if not self.distillation_enabled:
            # Standard training without distillation
            print("📚 Running STUDENT-ONLY training (no distillation)")
            return self.student.loss(batch_inputs_dict, batch_data_samples)
        
        print("🎓 Running KNOWLEDGE DISTILLATION training")
        
        try:
            # Clear GPU cache before processing
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get student outputs and features
            student_outputs, student_features = self.student_wrapper(
                batch_inputs_dict, batch_data_samples, mode='loss'
            )
            
            # For teacher, we need to handle different voxel configurations
            # The teacher will re-process the raw point cloud data with its own config
            with torch.no_grad():
                try:
                    teacher_outputs, teacher_features = self.teacher_wrapper(
                        batch_inputs_dict, batch_data_samples, mode='loss'
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e) or "memory" in str(e).lower():
                        print(f"Error in teacher forward: {e}")
                        print("Warning: Teacher model failed, falling back to student-only training")
                        teacher_outputs = None
                        teacher_features = None
                    else:
                        raise e
            
            # If teacher failed, fall back to student-only training
            if teacher_outputs is None:
                print("Warning: Teacher model failed, falling back to student-only training")
                if isinstance(student_outputs, dict):
                    return {k: v for k, v in student_outputs.items() if 'loss' in k}
                else:
                    # Create a dummy loss dict if student_outputs is not a dict
                    return {'loss_bbox': torch.tensor(0.0, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')}
            
            # Compute distillation losses
            distill_losses = self.distillation_loss(
                student_features, teacher_features,
                student_outputs, teacher_outputs
            )
            
            # Combine student task losses with distillation losses
            total_losses = {}
            
            # Add student task losses
            if isinstance(student_outputs, dict):
                for key, loss in student_outputs.items():
                    if 'loss' in key:
                        total_losses[key] = loss
            
            # Add distillation losses with prefix
            for key, loss in distill_losses.items():
                total_losses[f'distill_{key}'] = loss
            
            return total_losses
            
        except Exception as e:
            print(f"Error in distillation loss computation: {e}")
            # Fall back to student-only training
            return self.student.loss(batch_inputs_dict, batch_data_samples)
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample]) -> List[Det3DDataSample]:
        """Predict using student model."""
        return self.student.predict(batch_inputs_dict, batch_data_samples)
    
    def forward(self, inputs: dict = None, data_samples: List[Det3DDataSample] = None, mode: str = 'tensor', **kwargs):
        """Forward function for different modes."""
        # Handle different parameter naming conventions
        batch_inputs_dict = inputs if inputs is not None else kwargs.get('batch_inputs_dict')
        batch_data_samples = data_samples if data_samples is not None else kwargs.get('batch_data_samples')
        
        if mode == 'loss':
            return self.loss(batch_inputs_dict, batch_data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs_dict, batch_data_samples)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def _forward(self, inputs: dict = None, data_samples: List[Det3DDataSample] = None, mode: str = 'tensor', **kwargs):
        """Internal forward function."""
        return self.forward(inputs, data_samples, mode, **kwargs)
    
    def extract_feat(self, batch_inputs_dict: dict):
        """Extract features using student model."""
        return self.student.extract_feat(batch_inputs_dict)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.student.train(mode)
        if self.teacher is not None:
            self.teacher.eval()  # Teacher always in eval mode
        return self 