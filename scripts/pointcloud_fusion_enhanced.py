#!/usr/bin/env python3
"""
Enhanced CPU-only point cloud fusion from DiLiGenT-MV depth + masks.
Improvements vs quick_pointcloud.py:
 - Uses more views and lower stride for density
 - Optional mask erosion to reduce border noise
 - Open3D Statistical/Radius outlier removal to clean the cloud
 - Optional voxel downsampling for uniformity

Outputs: PLY and a preview PNG scatter.
"""
import argparse
import os
import glob
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
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


def backproject_depth(depth, K, mask=None, stride=2):
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    if mask is not None:
        m = mask[::stride, ::stride] > 0
    else:
        m = np.ones_like(z, dtype=bool)

    ones = np.ones_like(xx, dtype=np.float32)
    pix = np.stack([xx.astype(np.float32), yy.astype(np.float32), ones], axis=-1)  # (h,w,3)
    K33 = K[:3, :3]
    Kinv = np.linalg.inv(K33)
    rays = (Kinv @ pix.reshape(-1, 3).T).T.reshape(*pix.shape)  # (h,w,3)
    pts_cam = rays * z[..., None]
    m_flat = m.reshape(-1)
    pts = pts_cam.reshape(-1, 3)[m_flat]
    return pts


def save_png_preview(pts_w, cams, out_png, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pts_w.shape[0] > 0:
        step = max(1, pts_w.shape[0] // 200000)  # cap points used for preview to keep file small
        ax.scatter(pts_w[::step, 0], pts_w[::step, 1], pts_w[::step, 2], s=0.3, c='black', alpha=0.6)
    if len(cams) > 0:
        cams = np.array(cams)
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c='red', s=25, label='cams')
        ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./data/diligent_mv_normals')
    ap.add_argument('--obj', type=str, default='bear')
    ap.add_argument('--views', type=int, default=-1, help='Use first N views; -1 for all')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--mask_erode', type=int, default=1)
    ap.add_argument('--voxel', type=float, default=0.0, help='Open3D voxel size in world units (0 to disable)')
    ap.add_argument('--stat_nb', type=int, default=30)
    ap.add_argument('--stat_std', type=float, default=1.5)
    ap.add_argument('--rad_radius', type=float, default=0.02)
    ap.add_argument('--rad_min', type=int, default=8)
    ap.add_argument('--out_dir', type=str, default='./exp/quick_outputs')
    args = ap.parse_args()

    obj_dir = os.path.join(args.data_root, args.obj)
    cam_path = os.path.join(obj_dir, 'cameras_sphere.npz')
    cams = np.load(cam_path)

    # Collect masks and depths using filename indices
    mask_files = sorted(glob.glob(os.path.join(obj_dir, 'mask', '*.png')))
    depth_files = sorted(glob.glob(os.path.join(obj_dir, 'integration', 'depth', '*.npy')))
    if len(mask_files) == 0 or len(depth_files) == 0:
        raise RuntimeError('Could not find masks or depth .npy files.')

    # Align by index from filenames
    indices = [int(os.path.splitext(os.path.basename(p))[0]) for p in mask_files]
    mask_pairs = list(zip(indices, mask_files))
    mask_pairs.sort()
    if args.views > 0:
        mask_pairs = mask_pairs[:args.views]

    os.makedirs(args.out_dir, exist_ok=True)

    all_pts = []
    cam_centers = []

    kernel = np.ones((args.mask_erode, args.mask_erode), np.uint8) if args.mask_erode > 0 else None

    for idx, mpath in mask_pairs:
        if f'world_mat_{idx}' not in cams or f'scale_mat_{idx}' not in cams:
            continue
        P = (cams[f'world_mat_{idx}'].astype(np.float32) @ cams[f'scale_mat_{idx}'].astype(np.float32))[:3, :4]
        intr, pose = decompose_P(P)
        K = intr[:3, :3]
        c2w = pose
        cam_centers.append(c2w[:3, 3])

        # Depth path with aligned index
        dpath = os.path.join(obj_dir, 'integration', 'depth', f'{idx:03d}.npy')
        if not os.path.exists(dpath):
            # fallback: matching by order (robust to naming mismatch)
            if len(depth_files) > idx:
                dpath = depth_files[idx]
            else:
                continue

        mask = cv.imread(mpath, cv.IMREAD_GRAYSCALE)
        if kernel is not None:
            mask = cv.erode(mask, kernel)
        depth = np.load(dpath).astype(np.float32)
        if depth.shape != mask.shape:
            H, W = mask.shape
            depth = cv.resize(depth, (W, H), interpolation=cv.INTER_NEAREST)

        pts_cam = backproject_depth(depth, intr, mask=mask, stride=args.stride)
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts_cam, ones], axis=1)
        pts_w = (c2w @ pts_h.T).T[:, :3]
        all_pts.append(pts_w)

    if len(all_pts) == 0:
        raise RuntimeError('No points fused; check depth/mask files.')

    pts = np.concatenate(all_pts, axis=0)

    # Clean with Open3D if available
    if HAS_O3D:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        if args.voxel > 0:
            pc = pc.voxel_down_sample(voxel_size=args.voxel)
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=args.stat_nb, std_ratio=args.stat_std)
        pc, _ = pc.remove_radius_outlier(nb_points=args.rad_min, radius=args.rad_radius)
        o3d.io.write_point_cloud(os.path.join(args.out_dir, f'{args.obj}_pc_enhanced.ply'), pc)
        pts = np.asarray(pc.points)
    else:
        # ASCII fallback
        ply_path = os.path.join(args.out_dir, f'{args.obj}_pc_enhanced.ply')
        with open(ply_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {pts.shape[0]}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('end_header\n')
            for p in pts:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')

    # PNG preview
    save_png_preview(pts, cam_centers, os.path.join(args.out_dir, f'{args.obj}_pc_enhanced.png'),
                     f'{args.obj}: enhanced fused point cloud')
    print('✓ Saved:', os.path.join(args.out_dir, f'{args.obj}_pc_enhanced.ply'))
    print('✓ Saved:', os.path.join(args.out_dir, f'{args.obj}_pc_enhanced.png'))


if __name__ == '__main__':
    main()
