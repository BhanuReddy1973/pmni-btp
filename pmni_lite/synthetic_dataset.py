"""
Simple synthetic multi-view dataset for a unit sphere.

This provides the same minimal API as `MultiViewDataset` used by the trainer:
- `get_camera_rays(idx)` -> rays_o, rays_d
- `sample_random_pixels(n_pixels, views=None)` -> batch dict with rays_o, rays_d, normals, mask, depth (optional)

The synthetic data is analytic (ray-sphere intersections and normals) so it runs on CPU
and is fast for quick demos on Review 2.
"""
import numpy as np
import torch


def _make_sphere_pose(theta, phi, radius=1.5):
    """Create camera-to-world pose looking at origin from spherical coords."""
    # Camera position in world
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    pos = np.array([x, y, z], dtype=np.float32)

    # Look at origin
    forward = -pos / np.linalg.norm(pos)
    # arbitrary up vector
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=1)  # world->cam? we want cam->world
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = pos
    return pose


class MultiViewSyntheticDataset:
    """Synthetic dataset of a unit sphere with analytic normals and depths."""
    def __init__(self, n_views=8, H=64, W=64, fov=60.0, device='cpu'):
        self.n_images = n_views
        self.H = H
        self.W = W
        self.device = torch.device(device)

        # Create camera intrinsics (simple pinhole)
        fy = fx = 0.5 * H / np.tan(np.deg2rad(fov * 0.5))
        cx = W / 2.0
        cy = H / 2.0
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        self.intrinsics_all = torch.from_numpy(np.stack([K] * n_views)).float()

        # Create camera poses on a ring around the object
        thetas = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
        poses = [_make_sphere_pose(t, 0.1) for t in thetas]
        self.pose_all = torch.from_numpy(np.stack(poses)).float()

        # Precompute ray grids per view
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)

        # Sphere radius
        self.radius = 0.5

    def get_camera_rays(self, idx, resolution_level=1):
        H, W = self.H // resolution_level, self.W // resolution_level

        tx = torch.linspace(0, self.W - 1, W)
        ty = torch.linspace(0, self.H - 1, H)
        pixels_y, pixels_x = torch.meshgrid(ty, tx, indexing='ij')

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # (H, W, 3)
        p = torch.matmul(self.intrinsics_all_inv[idx, :3, :3], p[..., None]).squeeze(-1)

        rays_d = p / torch.linalg.norm(p, dim=-1, keepdim=True)
        rays_d = torch.matmul(self.pose_all[idx, :3, :3], rays_d[..., None]).squeeze(-1)
        rays_o = self.pose_all[idx, :3, 3].expand(rays_d.shape)

        return rays_o.to(self.device), rays_d.to(self.device)

    def _ray_sphere_intersect(self, ro, rd, center=torch.tensor([0.0, 0.0, 0.0]), r=0.5):
        # ro, rd: (..., 3) tensors
        # returns depth (t) or inf for miss, mask
        oc = ro - center.to(ro.device)
        a = (rd * rd).sum(dim=-1)
        b = 2.0 * (oc * rd).sum(dim=-1)
        c = (oc * oc).sum(dim=-1) - r * r
        disc = b * b - 4 * a * c
        mask = disc >= 0
        t = torch.full_like(a, float('inf'))
        if mask.any():
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t0 = (-b - sqrt_disc) / (2 * a)
            t1 = (-b + sqrt_disc) / (2 * a)
            # choose nearest positive
            t_pos = torch.where(t0 > 1e-4, t0, t1)
            t = torch.where(t_pos > 1e-4, t_pos, float('inf'))
        return t, mask

    def sample_random_pixels(self, n_pixels, views=None):
        if views is None:
            views = list(range(self.n_images))

        view_ids = np.random.choice(views, n_pixels, replace=True)
        ys = np.random.randint(0, self.H, n_pixels)
        xs = np.random.randint(0, self.W, n_pixels)

        pixels_y = torch.from_numpy(ys).long()
        pixels_x = torch.from_numpy(xs).long()
        view_ids_t = torch.from_numpy(view_ids).long()

        # Precompute rays for all views once
        rays_o_all = []
        rays_d_all = []
        for v in range(self.n_images):
            ro, rd = self.get_camera_rays(v)
            rays_o_all.append(ro)
            rays_d_all.append(rd)
        rays_o_all = torch.stack(rays_o_all, dim=0)
        rays_d_all = torch.stack(rays_d_all, dim=0)

        rays_o = rays_o_all[view_ids_t, pixels_y, pixels_x].to(self.device)
        rays_d = rays_d_all[view_ids_t, pixels_y, pixels_x].to(self.device)

        # Intersect with sphere
        depth, mask = self._ray_sphere_intersect(rays_o, rays_d, r=self.radius)
        mask = mask.float()

        # Compute surface points and normals where mask
        surf_pts = rays_o + depth.unsqueeze(-1) * rays_d
        normals = torch.nn.functional.normalize(surf_pts, dim=-1)

        batch = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'normals': normals,
            'mask': mask,
            'view_ids': view_ids_t,
            'depth': depth,
        }
        return batch
