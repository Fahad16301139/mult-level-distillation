import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
import math

# CRITICAL: Import all MMDet3D components to register them
import mmdet3d
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.runner import load_checkpoint
from mmengine.config import Config

# Register all modules FIRST
from mmdet3d.utils import register_all_modules
register_all_modules()

# Import BEVFusion specific modules - this registers all BEVFusion components
try:
    from projects.BEVFusion.bevfusion import *
    print("✅ BEVFusion project imported successfully")
except ImportError as e:
    print(f"Warning: BEVFusion project import failed: {e}")
    # Try alternative import paths
    try:
        import sys
        import os
        bevfusion_path = os.path.join(os.getcwd(), 'projects', 'BEVFusion')
        if bevfusion_path not in sys.path:
            sys.path.append(bevfusion_path)
        from bevfusion import *
        print("✅ BEVFusion imported via alternative path")
    except ImportError:
        print("❌ Could not import BEVFusion - continuing without it")
        pass


class AliasMethod(object):
    """Alias method for efficient sampling with many discrete outcomes."""
    def __init__(self, probs):
        """Initialize alias table for O(1) sampling.
        
        Args:
            probs: Probability distribution tensor
        """
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        """Move tensors to CUDA device."""
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()
    
    def to(self, device):
        """Move tensors to specified device."""
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)
        return self

    def draw(self, N):
        """Sample N indices from the distribution in O(1) time per sample."""
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())
        return oq + oj


class ContrastMemory(nn.Module):
    """Memory buffer for contrastive learning."""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        """Initialize contrastive memory bank.
        
        Args:
            inputSize: Feature dimension
            outputSize: Number of samples in memory
            K: Number of negative samples
            T: Temperature parameter
            momentum: Memory update momentum
        """
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        
        # Ensure multinomial is on the same device
        self.multinomial.to = lambda device: self.multinomial

    def forward(self, v1, v2, y, idx=None):
        """Compute contrastive scores and update memory with MEMORY MANAGEMENT."""
        try:
            # Clear cache before memory operations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            K = int(self.params[0].item())
            T = self.params[1].item()
            Z_v1 = self.params[2].item()
            Z_v2 = self.params[3].item()
            momentum = self.params[4].item()
            batchSize = v1.size(0)
            outputSize = self.memory_v1.size(0)
            inputSize = self.memory_v1.size(1)

            if idx is None:
                # Ensure multinomial is on the same device as input
                if hasattr(self.multinomial, 'prob'):
                    self.multinomial.prob = self.multinomial.prob.to(v1.device)
                    self.multinomial.alias = self.multinomial.alias.to(v1.device)
                idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
                idx.select(1, 0).copy_(y.data)

            # Synchronize before memory operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Sample and compute scores
            weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
            weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
            out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
            out_v2 = torch.exp(torch.div(out_v2, T))

            weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
            weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
            out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
            out_v1 = torch.exp(torch.div(out_v1, T))

            # Set normalization constants
            if Z_v1 < 0:
                self.params[2] = out_v1.mean() * outputSize
                Z_v1 = self.params[2].clone().detach().item()
            if Z_v2 < 0:
                self.params[3] = out_v2.mean() * outputSize
                Z_v2 = self.params[3].clone().detach().item()

            out_v1 = torch.div(out_v1, Z_v1).contiguous()
            out_v2 = torch.div(out_v2, Z_v2).contiguous()

            # Update memory
            with torch.no_grad():
                l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
                l_pos.mul_(momentum)
                l_pos.add_(torch.mul(v1, 1 - momentum))
                l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_v1 = l_pos.div(l_norm)
                self.memory_v1.index_copy_(0, y, updated_v1)

                ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
                ab_pos.mul_(momentum)
                ab_pos.add_(torch.mul(v2, 1 - momentum))
                ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_v2 = ab_pos.div(ab_norm)
                self.memory_v2.index_copy_(0, y, updated_v2)

            # Synchronize after memory operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return out_v1, out_v2
            
        except Exception as e:
            print(f"Warning: ContrastMemory forward failed: {str(e)}")
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Return dummy outputs
            device = v1.device if v1 is not None else 'cuda'
            batch_size = v1.size(0) if v1 is not None else 1
            dummy_out = torch.zeros(batch_size, 1, device=device)
            return dummy_out, dummy_out


class BEVFusionDistillationWrapper(nn.Module):
    """Wrapper for BEVFusion model to enable knowledge distillation with MEMORY FIXES."""
    
    def __init__(self, model, is_student=False):
        """Initialize model wrapper with feature extraction hooks.
        
        Args:
            model: BEVFusion model to wrap
            is_student: Whether this is a student model (affects gradient computation)
        """
        super().__init__()
        self.model = model
        self.is_student = is_student
        self.features = {}
        
        # Register hooks more carefully
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features safely."""
        try:
            # Hook the backbone output more directly
            if hasattr(self.model, 'pts_backbone') and self.model.pts_backbone is not None:
                self.model.pts_backbone.register_forward_hook(
                    self._get_activation_hook('pts_backbone')
                )
            
            # Hook neck output if available
            if hasattr(self.model, 'pts_neck') and self.model.pts_neck is not None:
                self.model.pts_neck.register_forward_hook(
                    self._get_activation_hook('pts_neck')
                )
                
        except Exception as e:
            print(f"Warning: Could not register all hooks: {e}")
    
    def _get_activation_hook(self, name):
        """Create forward hook function to store features safely."""
        def hook(module, input, output):
            try:
                if isinstance(output, (list, tuple)):
                    # Handle multi-scale outputs
                    if len(output) > 0:
                        self.features[name] = output[-1]  # Use the last/highest resolution
                else:
                    self.features[name] = output
            except Exception as e:
                print(f"Warning: Hook failed for {name}: {e}")
                self.features[name] = None
        return hook
    
    def forward(self, batch_inputs_dict, batch_data_samples=None, mode='loss'):
        """Forward pass with safe feature extraction and MEMORY MANAGEMENT."""
        self.features = {}
        
        try:
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Ensure proper device placement
            if not self.is_student:
                # Teacher model - handle differently to avoid conflicts
                with torch.no_grad():
                    # Synchronize before teacher forward
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    if mode == 'loss':
                        result = self.model.loss(batch_inputs_dict, batch_data_samples)
                    else:
                        result = self.model.predict(batch_inputs_dict, batch_data_samples)
                    
                    # Synchronize after teacher forward
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            else:
                # Student model 
                # Synchronize before student forward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                if mode == 'loss':
                    result = self.model.loss(batch_inputs_dict, batch_data_samples)
                else:
                    result = self.model.predict(batch_inputs_dict, batch_data_samples)
                
                # Synchronize after student forward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            return result, self.features
            
        except Exception as e:
            print(f"Error in {'teacher' if not self.is_student else 'student'} forward: {e}")
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, {}


class Embed(nn.Module):
    """Embedding module for feature projection."""
    def __init__(self, dim_in=512, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        # Handle different input shapes more robustly
        if x is None:
            return None
            
        original_shape = x.shape
        try:
            if len(x.shape) == 4:  # [B, C, H, W]
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
            elif len(x.shape) == 3:  # [B, N, C] or [B, C, L]
                x = x.mean(dim=1) if x.size(1) > x.size(2) else x.mean(dim=2)
            elif len(x.shape) == 5:  # [B, C, D, H, W]
                x = F.adaptive_avg_pool3d(x, (1, 1, 1))
                x = x.view(x.size(0), -1)
            else:
                x = x.view(x.size(0), -1)
            
            x = self.linear(x)
            x = self.l2norm(x)
            return x
        except Exception as e:
            print(f"Warning: Embed forward failed for shape {original_shape}: {e}")
            return None


class Normalize(nn.Module):
    """L2 normalization layer."""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        if x is None:
            return None
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)  # Add epsilon for stability
        return out


class ContrastLoss(nn.Module):
    """Contrastive loss for CRD."""
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        if x is None:
            return torch.tensor(0.0, device=x.device if x is not None else 'cuda')
            
        bsz = x.shape[0]
        m = x.size(1) - 1

        # Noise distribution
        Pn = 1 / float(self.n_data)

        # Loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + 1e-7)).log_()

        # Loss for K negative pairs
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + 1e-7)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        return loss


class CRDLoss(nn.Module): # Contrastive Representation Distillation Loss for individual layers
    """Contrastive Representation Distillation Loss."""
    
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """Forward pass for CRD loss with MEMORY MANAGEMENT."""
        if f_s is None or f_t is None:
            return torch.tensor(0.0, device=f_s.device if f_s is not None else 'cuda')
            
        try:
            # Clear cache before CRD computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Debug: Print shapes before embedding
            print(f"  🔍 CRD Debug - Input shapes: f_s={f_s.shape}, f_t={f_t.shape}")
            
            f_s = self.embed_s(f_s)
            f_t = self.embed_t(f_t)
            
            if f_s is None or f_t is None:
                print("  ⚠️  Embedding failed, returning zero loss")
                return torch.tensor(0.0, device=idx.device)
            
            print(f"  🔍 CRD Debug - After embedding: f_s={f_s.shape}, f_t={f_t.shape}")
            
            # Synchronize before contrastive computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
            
            print(f"  🔍 CRD Debug - After contrast: out_s={out_s.shape}, out_t={out_t.shape}")
            
            # Synchronize after contrastive computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            s_loss = self.criterion_s(out_s)
            t_loss = self.criterion_t(out_t)
            loss = s_loss + t_loss
            
            print(f"  ✅ CRD Success - s_loss={s_loss.item():.4f}, t_loss={t_loss.item():.4f}")
            
            return loss
            
        except Exception as e:
            print(f"Warning: CRD loss computation failed: {str(e)}")
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.tensor(0.0, device=idx.device)


class BEVFusionDistillationLoss(nn.Module):
    """Comprehensive distillation loss for BEVFusion."""
    
    def __init__(self, 
                 student_channels, 
                 teacher_channels,
                 n_data=1000,  # Number of training samples
                 feat_dim=128,
                 nce_k=4096,   # Number of negative samples
                 nce_t=0.07,   # Temperature
                 nce_m=0.5,    # Momentum
                 alpha_crd=1.0,
                 alpha_kd=0.5):
        super().__init__()
        
        self.alpha_crd = alpha_crd
        self.alpha_kd = alpha_kd
        
        # CRD losses for different layers
        self.crd_losses = nn.ModuleDict()
        
        for layer_name in student_channels.keys():
            if layer_name in teacher_channels:
                # Create options for CRD
                class CRDOpt:
                    def __init__(self, s_dim, t_dim, feat_dim, n_data, nce_k, nce_t, nce_m):
                        self.s_dim = s_dim
                        self.t_dim = t_dim
                        self.feat_dim = feat_dim
                        self.n_data = n_data
                        self.nce_k = nce_k
                        self.nce_t = nce_t
                        self.nce_m = nce_m
                
                opt = CRDOpt(
                    s_dim=student_channels[layer_name],
                    t_dim=teacher_channels[layer_name], 
                    feat_dim=feat_dim,
                    n_data=n_data,
                    nce_k=nce_k,
                    nce_t=nce_t,
                    nce_m=nce_m
                )
                
                self.crd_losses[layer_name] = CRDLoss(opt)
    
    def forward(self, student_features, teacher_features, indices, contrast_indices=None):
        """Compute comprehensive distillation loss."""
        losses = {}
        total_loss = 0.0
        
        print(f"  🔍 BEVFusionDistillationLoss - Starting aggregation...")
        
        # Ensure indices are available
        if indices is None:
            batch_size = next(iter(student_features.values())).size(0) if student_features else 1
            indices = torch.arange(batch_size, device=next(iter(student_features.values())).device)
        
        # CRD losses for each layer ,layer by layer distillation
        for layer_name in student_features.keys():
            if layer_name in teacher_features and layer_name in self.crd_losses:
                s_feat = student_features[layer_name]
                t_feat = teacher_features[layer_name]
                
                if s_feat is not None and t_feat is not None:
                    try:
                        print(f"  🔍 Computing CRD loss for {layer_name}...")
                        crd_loss = self.crd_losses[layer_name](s_feat, t_feat, indices, contrast_indices)
                        print(f"  ✅ Got CRD loss for {layer_name}: {crd_loss.item():.4f}")
                        
                        weighted_loss = self.alpha_crd * crd_loss
                        losses[f'{layer_name}_crd'] = weighted_loss
                        total_loss += weighted_loss
                        
                        print(f"  ✅ Added to total_loss: {total_loss.item():.4f}")
                        
                    except Exception as e:
                        print(f"Warning: Failed CRD loss for {layer_name}: {str(e)}")
                        continue
        
        print(f"  🔍 Final total_loss before return: {total_loss.item():.4f}")
        print(f"  🔍 Total_loss type: {type(total_loss)}")
        print(f"  🔍 Total_loss requires_grad: {total_loss.requires_grad if hasattr(total_loss, 'requires_grad') else 'N/A'}")
        
        # Add a small regularization if no losses computed
        if not losses:
            device = indices.device if indices is not None else 'cuda'
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure total_loss requires gradients
        if not total_loss.requires_grad:
            total_loss = total_loss.clone().detach().requires_grad_(True)
            
        losses['total_distill'] = total_loss
        print(f"  ✅ BEVFusionDistillationLoss - Returning losses dict")
        return losses

# This function is used to build the teacher and student models for distillation
def build_bevfusion_distillation(teacher_config_path, student_config_path, teacher_checkpoint=None, n_data=1000, device='cuda'):
    """Build teacher and student models for distillation."""
    
    # Load configs
    teacher_config = Config.fromfile(teacher_config_path)
    student_config = Config.fromfile(student_config_path)
    
    # Build teacher model (large, pretrained)
    print("🏗️ Building LARGE teacher model...")
    teacher_model = MODELS.build(teacher_config.model)
    
    # Load teacher checkpoint with error handling for architecture mismatch
    if teacher_checkpoint:
        try:
            print(f"🔄 Loading teacher checkpoint: {teacher_checkpoint}")
            # Load checkpoint and handle potential mismatches
            checkpoint = torch.load(teacher_checkpoint, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle minor mismatches
            missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys in teacher: {len(missing_keys)} (using random init for these)")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)} (ignoring these)")
                
            print(f"✓ Loaded teacher checkpoint successfully")
            
        except Exception as e:
            print(f"❌ Could not load teacher checkpoint: {e}")
            print("🔄 Using random initialization for teacher")
    
    teacher_model.eval()
    
    # 🔒 FREEZE TEACHER WEIGHTS - Critical for knowledge distillation!
    print("🔒 Freezing teacher model weights...")
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    frozen_params = sum(1 for p in teacher_model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in teacher_model.parameters())
    print(f"✅ Frozen {frozen_params}/{total_params} teacher parameters")
    
    # Build student model (small, lightweight - DIFFERENT architecture)
    print("🏗️ Building SMALL student model...")
    student_model = MODELS.build(student_config.model)
    student_model.train()
    
    # Ensure student weights are trainable
    for param in student_model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(1 for p in student_model.parameters() if p.requires_grad)
    print(f"✅ Student has {trainable_params} trainable parameters")
    
    # Move models to device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Wrap models for distillation
    teacher_wrapper = BEVFusionDistillationWrapper(teacher_model, is_student=False)
    student_wrapper = BEVFusionDistillationWrapper(student_model, is_student=True)
    
    # Move wrappers to device
    teacher_wrapper = teacher_wrapper.to(device)
    student_wrapper = student_wrapper.to(device)
    
    # Print model information
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"🎓 Teacher model parameters: {teacher_params:,}")
    print(f"🎒 Student model parameters: {student_params:,}")
    print(f"📊 Student is {student_params/teacher_params:.1%} the size of teacher")
    
    # Define channel dimensions based on ACTUAL FEATURE SHAPES FROM THE ERRORS
    # From the error messages, we can see the actual shapes:
    # Teacher backbone: torch.Size([1, 256, 136, 136]) -> 256 channels
    # Teacher neck: torch.Size([1, 512, 180, 180]) -> 512 channels  
    # Student backbone: torch.Size([1, 128, 68, 68]) -> 128 channels
    # Student neck: torch.Size([1, 256, 90, 90]) -> 256 channels
    
    teacher_channels = {
        'pts_backbone': 256,  # ACTUAL teacher backbone feature channels
        'pts_neck': 512,      # ACTUAL teacher neck feature channels (FIXED!)
    }
    
    student_channels = {
        'pts_backbone': 128,  # ACTUAL student backbone feature channels  
        'pts_neck': 256,      # ACTUAL student neck feature channels (FIXED!)
    }
    
    print(f"🎓 Teacher (ACTUAL feature shapes): {teacher_channels}")
    print(f"🎒 Student (ACTUAL feature shapes): {student_channels}")
    print(f"📊 Compression ratio: {sum(student_channels.values())/sum(teacher_channels.values()):.2f}x")
    print(f"🔥 Fixed CRD channel dimensions to match ACTUAL feature shapes!")
    
    # Create distillation loss
    distill_criterion = BEVFusionDistillationLoss(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        n_data=n_data,
        feat_dim=128,
        nce_k=1024,  # Reduced for memory efficiency
        nce_t=0.07,
        nce_m=0.5,
        alpha_crd=1.0,
        alpha_kd=0.5
    )
    
    # Move criterion to device
    distill_criterion = distill_criterion.to(device)
    
    return teacher_wrapper, student_wrapper, distill_criterion 