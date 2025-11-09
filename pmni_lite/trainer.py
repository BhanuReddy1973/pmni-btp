"""
Trainer for PMNI-lite.
Handles training loop, checkpointing, logging, and mesh extraction.
"""
import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm


from .losses import PMNILiteLoss


class Trainer:
    """
    Training manager for PMNI-lite SDF reconstruction.
    """
    def __init__(self,
                 sdf_network,
                 renderer,
                 dataset,
                 exp_dir='exp/pmni_lite',
                 device='cpu',
                 lr=5e-4,
                 **loss_weights):
        
        self.sdf_network = sdf_network.to(device)
        self.renderer = renderer
        self.dataset = dataset
        self.device = torch.device(device)
        self.exp_dir = exp_dir
        
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'meshes'), exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(self.sdf_network.parameters(), lr=lr)
        
        # Loss function
        self.criterion = PMNILiteLoss(**loss_weights)
        
        self.iter = 0
        
    def train_step(self, batch_size=512):
        """
        Single training step.
        """
        self.sdf_network.train()
        
        # Sample batch
        batch = self.dataset.sample_random_pixels(batch_size)
        rays_o = batch['rays_o'].to(self.device)
        rays_d = batch['rays_d'].to(self.device)
        gt_normals = batch['normals'].to(self.device)
        gt_mask = batch['mask'].to(self.device)
        
        # Render
        render_out = self.renderer.render_batch(rays_o, rays_d)

        # Build surface points (depth may be zero for rays that didn't converge)
        surf_pts = rays_o + render_out['depth'].unsqueeze(-1) * rays_d

        # Only keep points where renderer found a surface AND GT mask says there is a surface.
        # This avoids using non-converged rays (depth==0 -> camera origin) which produce
        # bogus gradients / normals and lead to many floating fragments in the reconstructed mesh.
        pred_mask = (render_out['mask'] > 0.5).squeeze(-1)
        gt_mask_bool = (gt_mask > 0.5).squeeze(-1)
        valid_mask = pred_mask & gt_mask_bool

        # We'll use two kinds of valid surface samples:
        #  - rays where the renderer converged AND GT says surface (use renderer depth)
        #  - rays where the renderer did NOT converge but GT has depth (use GT depth)
        outputs = None
        targets = None

        has_gt_depth = 'depth' in batch
        gt_depth = batch['depth'].to(self.device).squeeze(-1) if has_gt_depth else None

        valid_pred = pred_mask & gt_mask_bool
        valid_gt_only = (~pred_mask) & gt_mask_bool if not has_gt_depth else ((~pred_mask) & gt_mask_bool & (~torch.isnan(gt_depth)))

        # Collect surface points and matching targets from both sets
        pts_list = []
        depth_list = []
        normal_tgt_list = []

        if valid_pred.any():
            pts_list.append(surf_pts[valid_pred])
            depth_list.append(render_out['depth'].squeeze(-1)[valid_pred])
            normal_tgt_list.append(gt_normals[valid_pred])

        if has_gt_depth and valid_gt_only.any():
            surf_pts_gt = rays_o + gt_depth.unsqueeze(-1) * rays_d
            pts_list.append(surf_pts_gt[valid_gt_only])
            depth_list.append(gt_depth[valid_gt_only])
            normal_tgt_list.append(gt_normals[valid_gt_only])

        if len(pts_list) > 0:
            surf_pts_valid = torch.cat(pts_list, dim=0)
            surf_pts_valid.requires_grad_(True)
            sdf_val, gradients = self.sdf_network.forward_with_nablas(surf_pts_valid)

            depths_concat = torch.cat(depth_list, dim=0)
            normals_tgt_concat = torch.cat(normal_tgt_list, dim=0)

            outputs = {
                'sdf_surface': sdf_val.squeeze(-1),
                'normal': gradients,
                'mask': torch.ones_like(sdf_val.squeeze(-1), device=self.device),
                'depth': depths_concat,
                'gradients': gradients,
            }

            targets = {
                'normal': normals_tgt_concat,
                'mask': torch.ones_like(sdf_val.squeeze(-1), dtype=torch.float32, device=self.device),
            }

            if has_gt_depth:
                # If GT depth exists, use the matching subset as target depth
                targets['depth'] = depths_concat

            # Compute loss using collected surface samples
            losses = self.criterion(outputs, targets)
        else:
            # No valid surface hits in this batch — return zero-like losses and skip the update.
            losses = {'sdf': 0.0, 'eikonal': 0.0, 'normal': 0.0, 'mask': 0.0, 'depth': 0.0, 'total': 0.0}
        
        # Backward (only if we have a tensor total loss). If losses are zero-like floats
        # (no valid surface samples), skip the gradient step to avoid calling .backward()
        self.optimizer.zero_grad()
        total_loss = losses['total']
        if torch.is_tensor(total_loss):
            total_loss.backward()
            self.optimizer.step()
        else:
            # losses['total'] is a float (no valid samples) — skip optimizer step
            pass
        
        self.iter += 1
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def train(self, n_iters=5000, batch_size=512, log_every=100, val_every=500, ckpt_every=1000):
        """
        Full training loop.
        """
        print(f'Starting training for {n_iters} iterations...')
        
        pbar = tqdm(range(n_iters))
        for i in pbar:
            losses = self.train_step(batch_size=batch_size)

            if log_every and log_every > 0 and (i + 1) % log_every == 0:
                loss_str = ' '.join([f'{k}={v:.4f}' for k, v in losses.items()])
                pbar.set_description(f'Iter {i+1}: {loss_str}')

            if ckpt_every and ckpt_every > 0 and (i + 1) % ckpt_every == 0:
                self.save_checkpoint(f'ckpt_{i+1:06d}.pth')

            if val_every and val_every > 0 and (i + 1) % val_every == 0:
                self.validate_and_save_mesh(f'mesh_{i+1:06d}.ply')
        
        # Final checkpoint and mesh
        self.save_checkpoint('final.pth')
        self.validate_and_save_mesh('final.ply')
        print('Training complete!')
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.exp_dir, 'checkpoints', filename)
        torch.save({
            'iter': self.iter,
            'model_state_dict': self.sdf_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        ckpt_path = os.path.join(self.exp_dir, 'checkpoints', filename)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter = checkpoint['iter']
        print(f'Checkpoint loaded: {ckpt_path}, iter={self.iter}')
    
    @torch.no_grad()
    def extract_mesh(self, resolution=128, bbox_min=(-0.6, -0.6, -0.1), bbox_max=(0.6, 0.6, 0.8), smooth_sigma=0.0):
        """
        Extract mesh using marching cubes.
        """
        self.sdf_network.eval()
        # Adaptive extraction: first evaluate on coarse grid to find region of interest,
        # then run marching cubes on a finer grid inside a tightened bbox. This helps
        # when the default bbox doesn't contain a zero crossing at the chosen resolution.

        # Helper: evaluate SDF on a set of points (Nx3) in chunks and return numpy array
        def eval_sdf_pts(pts_tensor):
            sdf_list = []
            chunk = 65536
            for i in range(0, pts_tensor.shape[0], chunk):
                p = pts_tensor[i:i+chunk].to(self.device)
                with torch.no_grad():
                    sdf_chunk = self.sdf_network(p).cpu().numpy()
                sdf_list.append(sdf_chunk)
            return np.concatenate(sdf_list, axis=0)

        # Coarse scan parameters
        coarse_res = min(32, max(8, resolution // 4))
        xs_c = np.linspace(bbox_min[0], bbox_max[0], coarse_res)
        ys_c = np.linspace(bbox_min[1], bbox_max[1], coarse_res)
        zs_c = np.linspace(bbox_min[2], bbox_max[2], coarse_res)
        grid_c = np.stack(np.meshgrid(xs_c, ys_c, zs_c, indexing='ij'), axis=-1).astype(np.float32)
        pts_c = torch.from_numpy(grid_c.reshape(-1, 3))

        sdf_c = eval_sdf_pts(pts_c).reshape(coarse_res, coarse_res, coarse_res)

        # Detect voxels near surface
        sdf_thresh = max(1e-3, 0.02 * max(np.ptp(xs_c), np.ptp(ys_c), np.ptp(zs_c)))
        near_mask = np.abs(sdf_c) <= sdf_thresh

        if near_mask.any():
            # Compute minimal bbox of voxels that are near surface
            idxs = np.argwhere(near_mask)
            mins = idxs.min(axis=0)
            maxs = idxs.max(axis=0)

            # Convert indices back to world coords
            vmin = np.array([xs_c[mins[0]], ys_c[mins[1]], zs_c[mins[2]]], dtype=np.float32)
            vmax = np.array([xs_c[maxs[0]], ys_c[maxs[1]], zs_c[maxs[2]]], dtype=np.float32)

            # Expand a bit to be safe
            pad = 1.2
            center = 0.5 * (vmin + vmax)
            half = 0.5 * (vmax - vmin)
            half = np.maximum(half, 0.01)
            half = half * pad
            bbox_min2 = (center - half).tolist()
            bbox_max2 = (center + half).tolist()
        else:
            # No near voxels found: center bbox around voxel with smallest |sdf|
            idx_flat = np.argmin(np.abs(sdf_c.reshape(-1)))
            idx = np.unravel_index(idx_flat, sdf_c.shape)
            center = np.array([xs_c[idx[0]], ys_c[idx[1]], zs_c[idx[2]]], dtype=np.float32)
            span = np.array(bbox_max) - np.array(bbox_min)
            # Choose a small bbox around the center (25% of original span)
            half = 0.25 * span
            bbox_min2 = (center - half).tolist()
            bbox_max2 = (center + half).tolist()

        # Now evaluate on fine grid inside bbox_min2/bbox_max2
        xs = np.linspace(bbox_min2[0], bbox_max2[0], resolution)
        ys = np.linspace(bbox_min2[1], bbox_max2[1], resolution)
        zs = np.linspace(bbox_min2[2], bbox_max2[2], resolution)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).astype(np.float32)
        pts = torch.from_numpy(grid.reshape(-1, 3))

        sdf_values = eval_sdf_pts(pts).reshape(resolution, resolution, resolution)

        # Optional smoothing of the SDF volume before marching cubes to suppress
        # high-frequency noise and grid artifacts. A small gaussian sigma (e.g. 0.5-1.5)
        # often improves the marching-cubes surface quality at the cost of tiny
        # blurring of sharp features. This requires scipy; if unavailable we skip.
        if smooth_sigma and smooth_sigma > 0.0:
            try:
                from scipy.ndimage import gaussian_filter
                sdf_values = gaussian_filter(sdf_values, sigma=float(smooth_sigma))
                print(f'Applied gaussian smoothing to SDF volume with sigma={smooth_sigma}')
            except Exception:
                print('scipy.ndimage.gaussian_filter not available; skipping SDF smoothing')

        # Marching cubes (import lazily to avoid requiring skimage at module import)
        try:
            from skimage import measure
        except ModuleNotFoundError:
            print('skimage not installed; skipping mesh extraction')
            return None

        try:
            verts, faces, normals, _ = measure.marching_cubes(sdf_values, level=0.0)
        except Exception:
            print('Marching cubes failed, no surface found')
            return None
        
        # Map to world coordinates
        vmin = np.array(bbox_min2, dtype=np.float32)
        vmax = np.array(bbox_max2, dtype=np.float32)
        verts_world = vmin + (vmax - vmin) * (verts / (resolution - 1))
        
        # Create Open3D mesh (import lazily)
        try:
            import open3d as o3d
        except ModuleNotFoundError:
            print('open3d not installed; cannot build mesh object')
            return None

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts_world.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        mesh.compute_vertex_normals()

        # Light smoothing of the mesh to reduce marching-cubes faceting
        try:
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        except Exception:
            # older open3d versions may not support this function; ignore
            pass

        return mesh
    
    def validate_and_save_mesh(self, filename):
        """Extract and save mesh."""
        mesh = self.extract_mesh()
        if mesh is not None:
            mesh_path = os.path.join(self.exp_dir, 'meshes', filename)
            try:
                import open3d as o3d
            except ModuleNotFoundError:
                print('open3d not installed; cannot save mesh')
                return

            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f'Mesh saved: {mesh_path}')
