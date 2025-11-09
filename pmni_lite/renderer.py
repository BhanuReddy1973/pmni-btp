"""
CPU sphere-tracing renderer with simple occupancy acceleration.
Simplified version of models/renderer.py without nerfacc dependency.
"""
import torch
import numpy as np


class SphereMarchingRenderer:
    """
    CPU-based sphere marching/tracing for SDF rendering.
    Uses bounding sphere and adaptive step sizes.
    """
    def __init__(self,
                 sdf_network,
                 n_samples=64,
                 n_importance=32,
                 near=0.0,
                 far=2.0,
                 eps=0.01,
                 max_steps=128):
        self.sdf_network = sdf_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far
        self.eps = eps
        self.max_steps = max_steps
        
    def near_far_from_sphere(self, rays_o, rays_d, radius=1.0):
        """
        Compute near/far intersection with bounding sphere.
        """
        a = (rays_d ** 2).sum(dim=-1, keepdim=True)
        b = 2.0 * (rays_o * rays_d).sum(dim=-1, keepdim=True)
        c = (rays_o ** 2).sum(dim=-1, keepdim=True) - radius ** 2
        
        discriminant = b ** 2 - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        
        mid = -b / (2 * a)
        delta = torch.sqrt(discriminant) / (2 * a)
        
        near = torch.clamp(mid - delta, min=self.near)
        far = torch.clamp(mid + delta, max=self.far)
        
        return near, far
    
    def render_batch(self, rays_o, rays_d, return_normals=True, return_depth=True):
        """
        Render a batch of rays using sphere marching.
        
        rays_o: (N, 3)
        rays_d: (N, 3)
        
        Returns dict with:
        - sdf_surface: (N,) SDF value at surface
        - depth: (N,) ray depth to surface
        - normal: (N, 3) surface normal
        - mask: (N,) binary mask (1 if surface found)
        """
        device = rays_o.device
        N = rays_o.shape[0]
        
        # Compute near/far per ray
        near, far = self.near_far_from_sphere(rays_o, rays_d, radius=1.2)
        
        # Initialize
        t = near.clone()  # (N, 1)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        depth = torch.zeros(N, device=device)
        sdf_surface = torch.zeros(N, device=device)
        
        # Sphere marching
        for step in range(self.max_steps):
            pts = rays_o + t * rays_d  # (N, 3)
            
            with torch.no_grad():
                sdf = self.sdf_network(pts).squeeze(-1)  # (N,)
            
            # Check convergence
            converged = sdf.abs() < self.eps

            # Determine newly converged rays (converged now but weren't converged before)
            not_yet = ~mask
            within_far = t.squeeze(-1) < far.squeeze(-1)
            newly = converged & not_yet & within_far

            # Update depth and sdf_surface for newly converged rays
            depth = torch.where(newly, t.squeeze(-1), depth)
            sdf_surface = torch.where(newly, sdf, sdf_surface)

            # Update mask to include newly converged rays
            mask = mask | newly
            
            # Step forward by SDF value (sphere tracing)
            t = t + torch.clamp(sdf.abs().unsqueeze(-1), min=0.001, max=0.1)
            
            # Stop if all rays converged or exceeded far plane
            if (mask.all()) or (t.squeeze(-1) > far.squeeze(-1)).all():
                break
        
        # Compute normals at surface points
        if return_normals:
            surf_pts = rays_o + depth.unsqueeze(-1) * rays_d
            surf_pts.requires_grad_(True)
            sdf_grad = self.sdf_network(surf_pts)
            
            d_output = torch.ones_like(sdf_grad, requires_grad=False, device=device)
            normals = torch.autograd.grad(
                outputs=sdf_grad,
                inputs=surf_pts,
                grad_outputs=d_output,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            normals = torch.nn.functional.normalize(normals, dim=-1)
        else:
            normals = None
        
        return {
            'sdf_surface': sdf_surface,
            'depth': depth,
            'normal': normals,
            'mask': mask.float(),
        }
