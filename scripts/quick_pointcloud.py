#!/usr/bin/env python3
"""
Quick CPU-only point cloud fusion from DiLiGenT-MV depth + masks.
Outputs:
- Fused point cloud .ply
- Optional camera poses plot .png
This provides a concrete visual artifact without running training.
"""
import argparse
import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


def decompose_P(P):
    import cv2 as cv
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def backproject_depth(depth, K, mask=None, stride=4):
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    if mask is not None:
        m = mask[::stride, ::stride] > 0
    else:
        m = np.ones_like(z, dtype=bool)
    # homogeneous pixels
    ones = np.ones_like(xx, dtype=np.float32)
    pix = np.stack([xx.astype(np.float32), yy.astype(np.float32), ones], axis=-1)  # (h,w,3)
    K33 = K[:3, :3]
    Kinv = np.linalg.inv(K33)
    rays = (Kinv @ pix.reshape(-1, 3).T).T.reshape(*pix.shape)  # (h,w,3)
    pts_cam = rays * z[..., None]
    m_flat = m.reshape(-1)
    pts = pts_cam.reshape(-1, 3)[m_flat]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./data/diligent_mv_normals', help='Dataset root')
    ap.add_argument('--obj', type=str, default='bear', help='Object name')
    ap.add_argument('--views', type=int, default=5, help='Number of first views to fuse')
    ap.add_argument('--stride', type=int, default=6, help='Pixel stride for subsampling')
    ap.add_argument('--out_dir', type=str, default='./exp/quick_outputs', help='Output directory')
    args = ap.parse_args()

    obj_dir = os.path.join(args.data_root, args.obj)
    cam_path = os.path.join(obj_dir, 'cameras_sphere.npz')
    cams = np.load(cam_path)

    world_mats = []
    scale_mats = []
    for i in range(args.views):
        world_mats.append(cams[f'world_mat_{i}'].astype(np.float32))
        scale_mats.append(cams[f'scale_mat_{i}'].astype(np.float32))

    mask_files = sorted(glob.glob(os.path.join(obj_dir, 'mask', '*.png')))[:args.views]
    depth_files = sorted(glob.glob(os.path.join(obj_dir, 'integration', 'depth', '*.npy')))[:args.views]

    if len(mask_files) == 0 or len(depth_files) == 0:
        raise RuntimeError('Could not find masks or depth .npy files.')

    os.makedirs(args.out_dir, exist_ok=True)

    all_pts = []
    cam_centers = []

    for i in range(min(args.views, len(mask_files), len(depth_files))):
        P = (world_mats[i] @ scale_mats[i])[:3, :4]
        intr, pose = decompose_P(P)
        K = intr[:3, :3]
        c2w = pose
        # camera center in world
        cam_centers.append(c2w[:3, 3])

        mask = cv.imread(mask_files[i], cv.IMREAD_GRAYSCALE)
        depth = np.load(depth_files[i]).astype(np.float32)
        if depth.shape != mask.shape:
            H, W = mask.shape
            depth = cv.resize(depth, (W, H), interpolation=cv.INTER_NEAREST)

        pts_cam = backproject_depth(depth, intr, mask=mask, stride=args.stride)
        # transform to world
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts_cam, ones], axis=1)  # (N,4)
        pts_w = (c2w @ pts_h.T).T[:, :3]
        all_pts.append(pts_w)

    if len(all_pts) == 0:
        raise RuntimeError('No points fused; check depth/mask files.')

    pts = np.concatenate(all_pts, axis=0)

    # Save as PLY
    ply_path = os.path.join(args.out_dir, f'{args.obj}_pc.ply')
    if HAS_O3D:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(ply_path, pc)
    else:
        # Simple ASCII PLY
        with open(ply_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {pts.shape[0]}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('end_header\n')
            for p in pts:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')

    # Camera plot
    cam_centers = np.array(cam_centers)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[::200,0], pts[::200,1], pts[::200,2], s=1, c='gray', alpha=0.4)
    ax.scatter(cam_centers[:,0], cam_centers[:,1], cam_centers[:,2], c='red', s=30, label='cams')
    ax.legend()
    ax.set_title(f'{args.obj}: fused point cloud (subsampled)')
    for axis in 'xyz':
        getattr(ax, f'set_{axis}label')(axis)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, f'{args.obj}_pc.png')
    fig.savefig(fig_path, dpi=150)
    print(f'✓ Saved: {ply_path}')
    print(f'✓ Saved: {fig_path}')

if __name__ == '__main__':
    main()
