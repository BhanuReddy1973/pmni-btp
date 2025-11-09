"""
Simplified DiLiGenT-MV dataset loader
"""

import torch
import numpy as np
import os
import cv2


class DiLiGentDataset:
    """Load DiLiGenT-MV dataset (simplified version)"""
    
    def __init__(self, data_dir, obj_name, device='cpu', max_views=None):
        """
        Args:
            data_dir: path to data/diligent_mv_normals
            obj_name: object name (e.g., 'bear')
            device: 'cpu' or 'cuda'
            max_views: limit number of views (for faster loading)
        """
        self.data_dir = data_dir
        self.obj_name = obj_name
        self.device = device
        
        obj_dir = os.path.join(data_dir, obj_name)
        
        # Load camera parameters
        camera_file = os.path.join(obj_dir, 'cameras_sphere.npz')
        cameras = np.load(camera_file)
        
        # Get camera matrices
        self.n_images = len([k for k in cameras.keys() if k.startswith('world_mat')])
        if max_views is not None:
            self.n_images = min(self.n_images, max_views)
        
        print(f"Loading {self.n_images} views...")
        
        # Load all data
        self.normals = []
        self.masks = []
        self.intrinsics_all = []
        self.c2w_all = []
        
        normal_dir = os.path.join(obj_dir, 'normal_camera_space_GT')
        mask_dir = os.path.join(obj_dir, 'mask')
        
        for idx in range(self.n_images):
            # Load normal map
            normal_path = os.path.join(normal_dir, f'{idx:03d}.png')
            normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            if normal_img is None:
                raise FileNotFoundError(f"Normal map not found: {normal_path}")
            
            # Convert from uint16 to float
            normal = normal_img.astype(np.float32) / 65535.0 * 2.0 - 1.0
            normal = torch.from_numpy(normal)  # (H, W, 3)
            
            # Load mask
            mask_path = os.path.join(mask_dir, f'{idx:03d}.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = torch.from_numpy(mask_img.astype(np.float32) / 255.0)  # (H, W)
            
            # Get camera matrices
            world_mat = cameras[f'world_mat_{idx}']  # (4, 4)
            scale_mat = cameras[f'scale_mat_{idx}']  # (4, 4)
            
            # P = world_mat @ scale_mat = K @ [R|t]
            P = world_mat @ scale_mat
            
            # Extract intrinsics and extrinsics
            # P = K @ [R|t], decompose using RQ decomposition
            K, R = self._decompose_projection_matrix(P[:3, :4])
            
            # Camera-to-world
            t = -R.T @ P[:3, 3:4]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3:4] = t
            
            self.normals.append(normal)
            self.masks.append(mask)
            self.intrinsics_all.append(torch.from_numpy(K.astype(np.float32)))
            self.c2w_all.append(torch.from_numpy(c2w.astype(np.float32)))
        
        self.H, self.W = self.normals[0].shape[:2]
        print(f"Image resolution: {self.H} x {self.W}")
    
    def _decompose_projection_matrix(self, P):
        """
        Decompose P = K[R|t] into intrinsics K and rotation R
        Using RQ decomposition
        """
        # RQ decomposition of first 3x3
        K, R = np.linalg.qr(P[:3, :3].T)
        K = K.T
        R = R.T
        
        # Make sure K has positive diagonal
        signs = np.sign(np.diag(K))
        K = K * signs[:, None]
        R = signs[:, None] * R
        
        # Normalize K
        K = K / K[2, 2]
        
        return K, R
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        """Get data for one view"""
        return {
            'normals': self.normals[idx],
            'mask': self.masks[idx],
            'intrinsics': self.intrinsics_all[idx],
            'c2w': self.c2w_all[idx],
            'idx': idx
        }


if __name__ == "__main__":
    # Quick test
    import sys
    
    data_dir = './data/diligent_mv_normals'
    obj_name = 'bear'
    
    if not os.path.exists(os.path.join(data_dir, obj_name)):
        print(f"Error: Dataset not found at {data_dir}/{obj_name}")
        sys.exit(1)
    
    print("Testing dataset loader...")
    dataset = DiLiGentDataset(data_dir, obj_name, max_views=5)
    
    print(f"\nDataset info:")
    print(f"  Number of views: {len(dataset)}")
    print(f"  Image size: {dataset.H} x {dataset.W}")
    
    # Test loading one view
    data = dataset[0]
    print(f"\nView 0:")
    print(f"  Normal shape: {data['normals'].shape}")
    print(f"  Mask shape: {data['mask'].shape}")
    print(f"  Intrinsics shape: {data['intrinsics'].shape}")
    print(f"  C2W shape: {data['c2w'].shape}")
    print(f"  Mask coverage: {data['mask'].mean():.2%}")
    
    print("\n✓ Dataset test passed!")
