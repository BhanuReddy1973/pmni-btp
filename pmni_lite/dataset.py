"""
Dataset loader for multi-view depth, normals, and masks.
Adapted from models/dataset_loader.py to be self-contained.
"""
import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
import pyexr


def load_K_Rt_from_P(P):
    """
    Decompose projection matrix P into intrinsics K and pose (c2w).
    From IDR/PMNI convention.
    """
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class MultiViewDataset:
    """
    Multi-view dataset for textureless objects.
    
    Expected structure:
    - data_dir/
      - cameras_sphere.npz (world_mat_i, scale_mat_i)
      - normal/*.exr (or normal_camera_space_GT/*.exr)
      - integration/depth/*.npy
      - mask/*.png
    """
    def __init__(self, 
                 data_dir,
                 normal_dir='normal',
                 depth_dir='integration/depth',
                 mask_dir='mask',
                 cameras_name='cameras_sphere.npz',
                 views=None,
                 device='cpu'):
        
        self.data_dir = data_dir
        self.device = torch.device(device)
        
        # Load camera parameters
        cam_path = os.path.join(data_dir, cameras_name)
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera file not found: {cam_path}")
        camera_dict = np.load(cam_path)
        self.camera_dict = camera_dict
        
        # Find available views
        normal_files = sorted(glob(os.path.join(data_dir, normal_dir, '*.exr')))
        depth_files = sorted(glob(os.path.join(data_dir, depth_dir, '*.npy')))
        mask_files = sorted(glob(os.path.join(data_dir, mask_dir, '*.png')))
        
        if len(normal_files) == 0:
            raise FileNotFoundError(f"No normal maps found in {os.path.join(data_dir, normal_dir)}")
        
        self.n_images = len(normal_files)
        self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in normal_files]
        
        # Select subset of views if specified
        if views is not None and views < self.n_images:
            normal_files = normal_files[:views]
            depth_files = depth_files[:views] if depth_files else []
            mask_files = mask_files[:views] if mask_files else []
            self.img_idx_list = self.img_idx_list[:views]
            self.n_images = views
        
        print(f'Loading {self.n_images} views from {data_dir}...')
        
        # Load normal maps
        print('Loading normal maps...')
        normal_np = np.stack([pyexr.read(f)[..., :3] for f in normal_files])
        # DiLiGenT convention: flip Y and Z for camera space
        normal_np[..., 1] *= -1
        normal_np[..., 2] *= -1
        self.normals = torch.from_numpy(normal_np.astype(np.float32))  # (N, H, W, 3)
        
        self.H, self.W = self.normals.shape[1], self.normals.shape[2]
        
        # Load depth maps if available
        if len(depth_files) >= self.n_images:
            print('Loading depth maps...')
            depth_np = np.stack([np.load(f) for f in depth_files[:self.n_images]])
            self.depths = torch.from_numpy(depth_np.astype(np.float32))  # (N, H, W)
        else:
            print('Depth maps not found, will be None')
            self.depths = None
        
        # Load masks
        print('Loading masks...')
        mask_np = np.stack([cv.imread(f, cv.IMREAD_GRAYSCALE) for f in mask_files[:self.n_images]]) / 255.0
        self.masks = torch.from_numpy(mask_np.astype(np.float32))  # (N, H, W)
        
        # Extract intrinsics and poses
        self.intrinsics_all = []
        self.pose_all = []
        
        for idx in self.img_idx_list:
            world_mat = camera_dict[f'world_mat_{idx}'].astype(np.float32)
            scale_mat = camera_dict[f'scale_mat_{idx}'].astype(np.float32)
            P = (world_mat @ scale_mat)[:3, :4]
            
            intrinsics, pose = load_K_Rt_from_P(P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        
        self.intrinsics_all = torch.stack(self.intrinsics_all)  # (N, 4, 4)
        self.pose_all = torch.stack(self.pose_all)  # (N, 4, 4)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        
        print(f'Dataset loaded: {self.n_images} views, resolution {self.H}x{self.W}')
        
    def get_camera_rays(self, idx, resolution_level=1):
        """
        Generate camera rays for view idx.
        Returns: rays_o (H, W, 3), rays_d (H, W, 3)
        """
        H, W = self.H // resolution_level, self.W // resolution_level
        
        tx = torch.linspace(0, self.W - 1, W)
        ty = torch.linspace(0, self.H - 1, H)
        pixels_y, pixels_x = torch.meshgrid(ty, tx, indexing='ij')
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # (H, W, 3)
        p = torch.matmul(self.intrinsics_all_inv[idx, :3, :3], p[..., None]).squeeze(-1)  # (H, W, 3)
        
        rays_d = p / torch.linalg.norm(p, dim=-1, keepdim=True)  # (H, W, 3)
        rays_d = torch.matmul(self.pose_all[idx, :3, :3], rays_d[..., None]).squeeze(-1)  # (H, W, 3)
        rays_o = self.pose_all[idx, :3, 3].expand(rays_d.shape)  # (H, W, 3)
        
        return rays_o, rays_d
    
    def sample_random_pixels(self, n_pixels, views=None):
        """
        Sample random pixels from specified views (or all views).
        Returns dict with rays, normals, masks, depths (if available).
        """
        if views is None:
            views = list(range(self.n_images))
        
        view_ids = np.random.choice(views, n_pixels, replace=True)
        ys = np.random.randint(0, self.H, n_pixels)
        xs = np.random.randint(0, self.W, n_pixels)
        
        # Gather data
        pixels_y = torch.from_numpy(ys).long()
        pixels_x = torch.from_numpy(xs).long()
        view_ids_t = torch.from_numpy(view_ids).long()
        
        # Get rays
        rays_o_all = []
        rays_d_all = []
        for v in range(self.n_images):
            ro, rd = self.get_camera_rays(v)
            rays_o_all.append(ro)
            rays_d_all.append(rd)
        rays_o_all = torch.stack(rays_o_all, dim=0)  # (N, H, W, 3)
        rays_d_all = torch.stack(rays_d_all, dim=0)
        
        rays_o = rays_o_all[view_ids_t, pixels_y, pixels_x]  # (n_pixels, 3)
        rays_d = rays_d_all[view_ids_t, pixels_y, pixels_x]
        normals = self.normals[view_ids_t, pixels_y, pixels_x]  # (n_pixels, 3)
        masks = self.masks[view_ids_t, pixels_y, pixels_x]  # (n_pixels,)
        
        batch = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'normals': normals,
            'mask': masks,
            'view_ids': view_ids_t,
        }
        
        if self.depths is not None:
            depths = self.depths[view_ids_t, pixels_y, pixels_x]
            batch['depth'] = depths
        
        return batch
