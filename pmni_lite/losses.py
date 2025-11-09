"""
Loss functions for PMNI-lite training.
Inspired by models/losses.py from PMNI.
"""
import torch
import torch.nn.functional as F


def sdf_loss(sdf_pred, target=0.0, mask=None):
    """
    Surface loss: enforce SDF = 0 on surface points.
    """
    loss = (sdf_pred - target).abs()
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean()


def eikonal_loss(gradients):
    """
    Eikonal constraint: |grad(sdf)| = 1
    """
    grad_norm = gradients.norm(2, dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()


def normal_consistency_loss(pred_normals, target_normals, mask=None):
    """
    Normal alignment loss: encourage predicted normals to match target.
    Uses cosine similarity.
    """
    pred_normals = F.normalize(pred_normals, dim=-1)
    target_normals = F.normalize(target_normals, dim=-1)
    
    # Cosine similarity (higher is better, range [-1, 1])
    cos_sim = (pred_normals * target_normals).sum(dim=-1)
    
    # Convert to loss (minimize 1 - cos_sim)
    loss = 1.0 - cos_sim
    
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean()


def mask_loss(pred_mask, target_mask):
    """
    Binary cross-entropy for predicted vs target mask.
    """
    return F.binary_cross_entropy(pred_mask.clamp(0, 1), target_mask, reduction='mean')


def depth_loss(pred_depth, target_depth, mask=None):
    """
    L1 loss on depth values.
    """
    loss = (pred_depth - target_depth).abs()
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.mean()


class PMNILiteLoss:
    """
    Combined loss for PMNI-lite training.
    """
    def __init__(self,
                 sdf_weight=1.0,
                 eikonal_weight=0.1,
                 normal_weight=1.0,
                 mask_weight=0.1,
                 depth_weight=0.5):
        self.sdf_weight = sdf_weight
        self.eikonal_weight = eikonal_weight
        self.normal_weight = normal_weight
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
    
    def __call__(self, outputs, targets):
        """
        Compute combined loss.
        
        outputs: dict with keys [sdf_surface, normal, mask, depth, gradients]
        targets: dict with keys [sdf, normal, mask, depth]
        """
        losses = {}
        total_loss = 0.0
        
        # SDF loss (on-surface constraint)
        if 'sdf_surface' in outputs and 'mask' in targets:
            mask_valid = targets['mask'] > 0.5
            loss_sdf = sdf_loss(outputs['sdf_surface'], target=0.0, mask=mask_valid)
            losses['sdf'] = loss_sdf
            total_loss += self.sdf_weight * loss_sdf
        
        # Eikonal loss
        if 'gradients' in outputs:
            loss_eik = eikonal_loss(outputs['gradients'])
            losses['eikonal'] = loss_eik
            total_loss += self.eikonal_weight * loss_eik
        
        # Normal consistency loss
        if 'normal' in outputs and 'normal' in targets and 'mask' in targets:
            mask_valid = targets['mask'] > 0.5
            loss_norm = normal_consistency_loss(outputs['normal'], targets['normal'], mask=mask_valid)
            losses['normal'] = loss_norm
            total_loss += self.normal_weight * loss_norm
        
        # Mask loss
        if 'mask' in outputs and 'mask' in targets:
            loss_mask = mask_loss(outputs['mask'], targets['mask'])
            losses['mask'] = loss_mask
            total_loss += self.mask_weight * loss_mask
        
        # Depth loss
        if 'depth' in outputs and 'depth' in targets and 'mask' in targets:
            mask_valid = targets['mask'] > 0.5
            loss_depth = depth_loss(outputs['depth'], targets['depth'], mask=mask_valid)
            losses['depth'] = loss_depth
            total_loss += self.depth_weight * loss_depth
        
        losses['total'] = total_loss
        return losses
