import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ProjectionMLP(nn.Module):
    """
    Multi-layer perceptron for projecting features to common embedding space.
    Adapted for BEV features which may have different spatial dimensions.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to embedding space.
        
        Args:
            x: Input features of various shapes
            
        Returns:
            Projected features normalized to unit sphere
        """
        # Handle different input shapes from BEVFusion layers
        original_shape = x.shape
        
        if len(x.shape) == 4:  # [B, C, H, W] - typical BEV features
            # Global average pooling for spatial features
            x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
            x = x.view(x.size(0), -1)  # [B, C]
        elif len(x.shape) == 3:  # [B, C, N] - point features or sequence
            x = x.mean(dim=2)  # [B, C]
        elif len(x.shape) == 2:  # [B, C] - already flattened
            pass
        else:
            # Flatten everything except batch dimension
            x = x.view(x.size(0), -1)
        
        # Project to embedding space and normalize
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class BEVContrastiveLoss(nn.Module):
    """
    Contrastive loss specifically designed for BEV features.
    Based on InfoNCE loss but adapted for the spatial nature of BEV representations.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between student and teacher features.
        
        Args:
            student_features: Normalized student features [B, D]
            teacher_features: Normalized teacher features [B, D]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = student_features.size(0)
        
        # Ensure features are normalized
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(student_features, teacher_features.T) / self.temperature
        
        # Labels are diagonal indices (positive pairs)
        labels = torch.arange(batch_size, device=student_features.device)
        
        # Symmetric contrastive loss
        loss_s2t = self.criterion(logits, labels)
        loss_t2s = self.criterion(logits.T, labels)
        
        return (loss_s2t + loss_t2s) / 2


class BEVAttentionTransfer(nn.Module):
    """
    Attention Transfer adapted for BEV spatial features.
    Transfers spatial attention patterns from teacher to student.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Transfer spatial attention from teacher to student.
        
        Args:
            student_feat: Student BEV features [B, C, H, W]
            teacher_feat: Teacher BEV features [B, C', H', W']
            
        Returns:
            Attention transfer loss
        """
        # Generate attention maps
        student_att = self._compute_attention_map(student_feat)
        teacher_att = self._compute_attention_map(teacher_feat)
        
        # Resize student attention to match teacher if necessary
        if student_att.shape[2:] != teacher_att.shape[2:]:
            student_att = F.interpolate(
                student_att, 
                size=teacher_att.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Compute MSE loss between attention maps
        loss = F.mse_loss(student_att, teacher_att)
        
        return loss
    
    def _compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map from feature tensor.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Attention map [B, 1, H, W]
        """
        # Sum of squares across channel dimension
        attention = features.pow(2).sum(1, keepdim=True)
        
        # Normalize to [0, 1]
        attention = F.normalize(attention.view(attention.size(0), -1), p=1, dim=1)
        attention = attention.view(features.size(0), 1, features.size(2), features.size(3))
        
        return attention


class BEVKnowledgeDistillation(nn.Module):
    """
    Standard knowledge distillation for classification outputs.
    Uses KL divergence between teacher and student logits.
    """
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher logits.
        
        Args:
            student_logits: Student model logits [B, num_classes]
            teacher_logits: Teacher model logits [B, num_classes]
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence and scale by temperature squared
        loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return loss


class ComprehensiveBEVDistillation(nn.Module):
    """
    Comprehensive distillation framework combining multiple distillation methods.
    Specifically designed for BEVFusion architecture.
    """
    
    def __init__(self, 
                 student_channels: Dict[str, int],
                 teacher_channels: Dict[str, int],
                 feat_dim: int = 128,
                 temperature: float = 0.07,
                 kd_temperature: float = 4.0):
        super().__init__()
        
        # Store channel configurations
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        
        # Feature projectors for different layers
        self.projectors = nn.ModuleDict()
        for layer_name in student_channels:
            if layer_name in teacher_channels:
                self.projectors[f'{layer_name}_student'] = ProjectionMLP(
                    student_channels[layer_name], out_dim=feat_dim
                )
                self.projectors[f'{layer_name}_teacher'] = ProjectionMLP(
                    teacher_channels[layer_name], out_dim=feat_dim
                )
        
        # Loss modules
        self.contrastive_loss = BEVContrastiveLoss(temperature)
        self.attention_loss = BEVAttentionTransfer()
        self.kd_loss = BEVKnowledgeDistillation(kd_temperature)
    
    def forward(self, 
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                student_outputs: Optional[Dict] = None,
                teacher_outputs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive distillation losses.
        
        Args:
            student_features: Dictionary of student intermediate features
            teacher_features: Dictionary of teacher intermediate features
            student_outputs: Student model outputs (optional)
            teacher_outputs: Teacher model outputs (optional)
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Feature-level distillation for each layer
        for layer_name in self.student_channels:
            if layer_name in student_features and layer_name in teacher_features:
                
                # Project features to common space
                student_proj = self.projectors[f'{layer_name}_student'](
                    student_features[layer_name]
                )
                teacher_proj = self.projectors[f'{layer_name}_teacher'](
                    teacher_features[layer_name]
                )
                
                # Contrastive loss
                losses[f'{layer_name}_contrastive'] = self.contrastive_loss(
                    student_proj, teacher_proj
                )
                
                # Attention transfer (for spatial features)
                if len(student_features[layer_name].shape) == 4:  # BEV features [B, C, H, W]
                    losses[f'{layer_name}_attention'] = self.attention_loss(
                        student_features[layer_name],
                        teacher_features[layer_name]
                    )
        
        # Logit distillation (if outputs provided)
        if student_outputs is not None and teacher_outputs is not None:
            # Handle different output formats from BEVFusion
            if isinstance(student_outputs, dict) and isinstance(teacher_outputs, dict):
                # Look for classification logits in the outputs
                for key in ['cls_score', 'logits', 'classification']:
                    if key in student_outputs and key in teacher_outputs:
                        losses['knowledge_distillation'] = self.kd_loss(
                            student_outputs[key], teacher_outputs[key]
                        )
                        break
        
        return losses


def get_feature_dimensions(model_config: Dict) -> Dict[str, int]:
    """
    Extract feature dimensions from BEVFusion model configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Dictionary mapping layer names to their output dimensions
    """
    dimensions = {}
    
    # Extract dimensions from config
    if 'pts_backbone' in model_config:
        backbone_config = model_config['pts_backbone']
        if 'out_channels' in backbone_config:
            # Sum all backbone output channels
            out_channels = backbone_config['out_channels']
            if isinstance(out_channels, list):
                dimensions['pts_backbone'] = sum(out_channels)
            else:
                dimensions['pts_backbone'] = out_channels
    
    if 'pts_neck' in model_config:
        neck_config = model_config['pts_neck']
        if 'out_channels' in neck_config:
            out_channels = neck_config['out_channels']
            if isinstance(out_channels, list):
                dimensions['pts_neck'] = sum(out_channels)
            else:
                dimensions['pts_neck'] = out_channels
    
    if 'pts_middle_encoder' in model_config:
        encoder_config = model_config['pts_middle_encoder']
        if 'encoder_channels' in encoder_config:
            # Get final encoder channels
            encoder_channels = encoder_config['encoder_channels']
            if isinstance(encoder_channels, (list, tuple)) and len(encoder_channels) > 0:
                # Get the last layer's output channels
                last_layer = encoder_channels[-1]
                if isinstance(last_layer, (list, tuple)):
                    dimensions['pts_middle_encoder'] = last_layer[-1]
                else:
                    dimensions['pts_middle_encoder'] = last_layer
    
    return dimensions 