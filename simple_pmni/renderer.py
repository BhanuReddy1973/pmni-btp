"""
Simplified volumetric renderer using sphere tracing (faster than nerfacc for small scenes)
"""

import torch
import torch.nn.functional as F
import numpy as np


def generate_rays(H, W, intrinsics, c2w):
    """
    Generate rays for a camera
    Args:
        H, W: image height and width
        intrinsics: (3, 3) camera intrinsics
        c2w: (4, 4) camera-to-world matrix
    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
    """
    device = intrinsics.device
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    
    # Pixel centers
    i = i.float() + 0.5
    j = j.float() + 0.5
    
    # Normalized image coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    dirs = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)
    
    # Transform to world space
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    
    return rays_o, rays_d


def sphere_tracing(sdf_fn, ray_o, ray_d, n_steps=64, near=0.5, far=2.0, threshold=5e-3):
    """
    Simple sphere tracing to find surface intersection
    Args:
        sdf_fn: function that takes (N, 3) points and returns (N, 1) SDF
        ray_o: (N, 3) ray origins  
        ray_d: (N, 3) ray directions (normalized)
        n_steps: max steps
        near, far: ray bounds
        threshold: surface threshold
    Returns:
        depth: (N,) depth values, -1 if no hit
        hit_mask: (N,) boolean mask of hits
    """
    device = ray_o.device
    N = ray_o.shape[0]
    
    t = torch.ones(N, device=device) * near
    hit_mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    for _ in range(n_steps):
        points = ray_o + t.unsqueeze(-1) * ray_d
        sdf = sdf_fn(points).squeeze(-1)
        
        # Hit if close to surface
        hit = (sdf.abs() < threshold) & (t < far)
        hit_mask |= hit
        
        # Stop if hit or outside bounds
        active = ~hit & (t < far)
        if not active.any():
            break
        
        # Step by SDF value
        t = t + sdf.abs() * 0.9  # 0.9 for safety
    
    depth = torch.where(hit_mask, t, torch.ones_like(t) * -1)
    return depth, hit_mask


def simple_volume_rendering(sdf_fn, ray_o, ray_d, n_samples=64, near=0.5, far=2.0):
    """
    Simplified volume rendering - sample uniformly along ray
    Args:
        sdf_fn: SDF network
        ray_o: (N, 3) ray origins
        ray_d: (N, 3) ray directions  
        n_samples: number of samples per ray
        near, far: ray bounds
    Returns:
        depth: (N,) rendered depth
        mask: (N,) opacity
        points_sampled: (N, n_samples, 3) sampled points for loss
    """
    device = ray_o.device
    N = ray_o.shape[0]
    
    # Uniform sampling
    t_vals = torch.linspace(0., 1., n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals  # (n_samples,)
    z_vals = z_vals.expand(N, n_samples)  # (N, n_samples)
    
    # Sample points
    points = ray_o.unsqueeze(1) + z_vals.unsqueeze(-1) * ray_d.unsqueeze(1)  # (N, n_samples, 3)
    
    # Get SDF values
    points_flat = points.reshape(-1, 3)
    sdf_flat = sdf_fn(points_flat).reshape(N, n_samples)
    
    # Convert SDF to density (simple sigmoid)
    alpha = torch.sigmoid(-sdf_flat * 50.0)  # (N, n_samples)
    
    # Volume rendering weights
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N, 1), device=device), 1. - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]  # (N, n_samples)
    
    weights = alpha * transmittance  # (N, n_samples)
    
    # Rendered depth and opacity
    depth = (weights * z_vals).sum(dim=-1)  # (N,)
    mask = weights.sum(dim=-1)  # (N,)
    
    return depth, mask, points


def render_normals(model, ray_o, ray_d, n_samples=64, near=0.5, far=2.0):
    """
    Render surface normals
    Returns:
        normals: (N, 3) rendered normals
        mask: (N,) opacity
    """
    device = ray_o.device
    N = ray_o.shape[0]
    
    # Sample points
    _, mask, points = simple_volume_rendering(
        model.sdf_network.forward, ray_o, ray_d, n_samples, near, far
    )
    
    # Get normals at sampled points
    points_flat = points.reshape(-1, 3)
    normals_flat = model.sdf_network.gradient(points_flat)
    normals_flat = F.normalize(normals_flat, dim=-1)
    normals = normals_flat.reshape(N, n_samples, 3)
    
    # Weight by volume rendering
    t_vals = torch.linspace(0., 1., n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(N, n_samples)
    
    points_flat = points.reshape(-1, 3)
    sdf_flat = model.sdf_network(points_flat).reshape(N, n_samples)
    alpha = torch.sigmoid(-sdf_flat * 50.0)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N, 1), device=device), 1. - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]
    weights = alpha * transmittance  # (N, n_samples)
    
    # Weighted average of normals
    rendered_normals = (weights.unsqueeze(-1) * normals).sum(dim=1)  # (N, 3)
    rendered_normals = F.normalize(rendered_normals, dim=-1)
    
    return rendered_normals, mask


if __name__ == "__main__":
    # Quick test
    print("Testing ray generation...")
    H, W = 128, 128
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = W / 2
    intrinsics[1, 2] = H / 2
    c2w = torch.eye(4)
    c2w[2, 3] = -2.0  # Move camera back
    
    rays_o, rays_d = generate_rays(H, W, intrinsics, c2w)
    print(f"✓ Rays: origins {rays_o.shape}, directions {rays_d.shape}")
    
    # Test rendering
    print("Testing sphere tracing...")
    def sphere_sdf(points):
        return points.norm(dim=-1, keepdim=True) - 0.5
    
    rays_o_flat = rays_o.reshape(-1, 3)[:1000]
    rays_d_flat = rays_d.reshape(-1, 3)[:1000]
    depth, hit = sphere_tracing(sphere_sdf, rays_o_flat, rays_d_flat)
    print(f"✓ Traced {hit.sum()} / {len(hit)} hits")
