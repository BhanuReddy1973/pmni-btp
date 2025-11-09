"""
Simplified PMNI Training Script
Lightweight version that can run on limited GPU or CPU
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import argparse

from model import SimplePMNI
from renderer import generate_rays, render_normals, simple_volume_rendering
from dataset import DiLiGentDataset


def eikonal_loss(gradients):
    """Eikonal regularization - gradients should have unit norm"""
    return ((gradients.norm(dim=-1) - 1.0) ** 2).mean()


def normal_loss(pred_normals, gt_normals, mask=None):
    """
    Normal supervision loss
    Args:
        pred_normals: (N, 3) predicted normals
        gt_normals: (N, 3) ground truth normals  
        mask: (N,) valid mask
    """
    # Cosine similarity loss
    cos_sim = (pred_normals * gt_normals).sum(dim=-1)
    cos_loss = (1.0 - cos_sim)
    
    # L1 loss
    l1_loss = (pred_normals - gt_normals).abs().mean(dim=-1)
    
    if mask is not None:
        cos_loss = (cos_loss * mask).sum() / (mask.sum() + 1e-6)
        l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-6)
    else:
        cos_loss = cos_loss.mean()
        l1_loss = l1_loss.mean()
    
    return cos_loss + 0.1 * l1_loss


def mask_loss(pred_mask, gt_mask):
    """Silhouette loss"""
    return F.binary_cross_entropy(pred_mask, gt_mask)


def train_simple_pmni(
    data_dir,
    obj_name,
    output_dir="./output",
    num_iterations=500,
    batch_size=512,
    lr=1e-3,
    device='cuda',
    hidden_dim=64,
    num_layers=4
):
    """
    Main training loop
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {obj_name}")
    dataset = DiLiGentDataset(data_dir, obj_name, device='cpu')  # Keep on CPU
    print(f"  {len(dataset)} views, resolution: {dataset.H}x{dataset.W}")
    
    # Create model
    print(f"Creating model: {hidden_dim}D x {num_layers} layers")
    model = SimplePMNI(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"Training for {num_iterations} iterations...")
    pbar = tqdm(range(num_iterations))
    
    for iter in pbar:
        # Sample random view
        view_idx = np.random.randint(0, len(dataset))
        data = dataset[view_idx]
        
        # Sample random pixels
        H, W = dataset.H, dataset.W
        num_rays = min(batch_size, H * W)
        
        pixel_indices = torch.randperm(H * W)[:num_rays]
        y_indices = pixel_indices // W
        x_indices = pixel_indices % W
        
        # Get GT data
        gt_normals = data['normals'].reshape(H * W, 3)[pixel_indices].to(device)
        gt_mask = data['mask'].reshape(H * W)[pixel_indices].to(device)
        
        # Generate rays
        intrinsics = data['intrinsics'].to(device)
        c2w = data['c2w'].to(device)
        
        # Create rays for sampled pixels only
        rays_o_full, rays_d_full = generate_rays(H, W, intrinsics, c2w)
        rays_o = rays_o_full.reshape(H * W, 3)[pixel_indices]
        rays_d = rays_d_full.reshape(H * W, 3)[pixel_indices]
        
        # Render
        pred_normals, pred_mask = render_normals(model, rays_o, rays_d, n_samples=32)
        
        # Sample points for eikonal loss
        _, _, points = simple_volume_rendering(
            model.sdf_network.forward, rays_o, rays_d, n_samples=16
        )
        gradients = model.sdf_network.gradient(points.reshape(-1, 3))
        
        # Compute losses
        loss_normal = normal_loss(pred_normals, gt_normals, gt_mask)
        loss_mask = mask_loss(pred_mask, gt_mask)
        loss_eik = eikonal_loss(gradients)
        
        # Total loss
        loss = loss_normal + 0.1 * loss_mask + 0.1 * loss_eik
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if iter % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'normal': f'{loss_normal.item():.4f}',
                'mask': f'{loss_mask.item():.4f}',
                'eik': f'{loss_eik.item():.4f}',
                'opacity': f'{pred_mask.mean().item():.3f}'
            })
        
        # Save checkpoint
        if (iter + 1) % 100 == 0 or iter == num_iterations - 1:
            checkpoint = {
                'iter': iter,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(output_dir, f'ckpt_{iter:05d}.pth'))
    
    print(f"\n✓ Training complete! Checkpoints saved to {output_dir}")
    return model


def extract_mesh(model, resolution=128, threshold=0.0, output_path='mesh.ply'):
    """
    Extract mesh using marching cubes
    """
    try:
        from skimage import measure
    except:
        print("Warning: scikit-image not found. Cannot extract mesh.")
        return
    
    print(f"Extracting mesh at resolution {resolution}...")
    device = next(model.parameters()).device
    
    # Create grid
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    
    # Evaluate SDF in batches
    batch_size = 100000
    sdf_values = []
    
    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size].to(device)
            sdf = model.sdf_network(batch)
            sdf_values.append(sdf.cpu())
    
    sdf_values = torch.cat(sdf_values, dim=0).reshape(resolution, resolution, resolution).numpy()
    
    # Marching cubes
    print("Running marching cubes...")
    vertices, faces, normals, _ = measure.marching_cubes(sdf_values, level=threshold)
    
    # Scale to [-1, 1]
    vertices = vertices / (resolution - 1) * 2 - 1
    
    # Save PLY
    print(f"Saving mesh to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"✓ Mesh saved: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/diligent_mv_normals')
    parser.add_argument('--obj_name', type=str, default='bear')
    parser.add_argument('--output_dir', type=str, default='./simple_output')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mesh_res', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Train
    model = train_simple_pmni(
        data_dir=args.data_dir,
        obj_name=args.obj_name,
        output_dir=args.output_dir,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Extract mesh
    mesh_path = os.path.join(args.output_dir, 'final_mesh.ply')
    extract_mesh(model, resolution=args.mesh_res, output_path=mesh_path)
