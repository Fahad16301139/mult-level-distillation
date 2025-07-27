import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

# Register all modules
register_all_modules()

# Import BEVFusion
try:
    from projects.BEVFusion.bevfusion import *
    print("BEVFusion imported successfully")
except ImportError as e:
    print(f"BEVFusion import failed: {e}")


def infonce(student_embed, teacher_embed, tau=0.07):
    """
    One-directional InfoNCE loss: student → teacher.
    Args:
        student_embed (Tensor): [B, C] Student BEV embeddings.
        teacher_embed (Tensor): [B, C] Teacher BEV embeddings.
        tau (float): Temperature scaling factor.
    """
    # Normalize embeddings
    student_embed = F.normalize(student_embed, dim=-1)
    teacher_embed = F.normalize(teacher_embed, dim=-1)

    # Compute similarity matrix [B, B]
    logits = torch.matmul(student_embed, teacher_embed.T) / tau
    labels = torch.arange(logits.size(0)).to(student_embed.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


class BEVFusionCLIPDistiller(nn.Module):
    def __init__(self, teacher_model, student_model, tau=0.07):
        super().__init__()
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = student_model.train()
        self.tau = tau
        self.student_proj = None
    def forward(self, batch_inputs_dict,data_samples):
        # Create proper batch_input_metas
        # Extract real metadata from data_samples
        
        # Try different access patterns
        if hasattr(data_samples[0], 'metainfo'):
            batch_input_metas = [sample.metainfo for sample in data_samples]
        elif hasattr(data_samples[0], 'meta'):
            batch_input_metas = [sample.meta for sample in data_samples]
        else:
            batch_input_metas = [{}] * len(data_samples)  # Empty fallback
        
        with torch.no_grad():
            teacher_bev = self.teacher.extract_feat(batch_inputs_dict, batch_input_metas)
        student_bev = self.student.extract_feat(batch_inputs_dict, batch_input_metas)

        # Handle case where extract_feat returns a list
        if isinstance(teacher_bev, list):
            teacher_bev = teacher_bev[0]  # Take first element
        if isinstance(student_bev, list):
            student_bev = student_bev[0]  # Take first element
        if self.student_proj is None: #we add student_proj to make sure the student and teacher have the same number of channels
            student_channels = student_bev.shape[1]
            teacher_channels = teacher_bev.shape[1]
            if student_channels != teacher_channels:
                self.student_proj=nn.Conv2d(student_channels, teacher_channels, kernel_size=1).cuda()
            else:
                self.student_proj = nn.Identity()
        student_bev_proj=self.student_proj(student_bev) # [B,128,H,w] to [B,256,h,w]
        teacher_embed = F.adaptive_avg_pool2d(teacher_bev, 1).flatten(1)
        student_embed = F.adaptive_avg_pool2d(student_bev_proj, 1).flatten(1)

        distill_loss = infonce(student_embed, teacher_embed, self.tau)
        return {"total_loss": distill_loss, "distill_loss": distill_loss}





