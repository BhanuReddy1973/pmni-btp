#!/usr/bin/env python3
"""
CPU-only Visual Hull baseline for DiLiGenT-MV style datasets.

Inputs:
  - data_dir: directory containing cameras_sphere.npz and mask/*.png
  - cameras_sphere.npz must have world_mat_i and scale_mat_i entries.

Outputs:
  - A coarse mesh (.ply) and a preview (.png) saved under --outdir

This avoids CUDA/nerfacc and runs quickly at low resolutions (e.g., 64^3).
"""

import argparse
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from skimage import measure
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_K_Rt_from_P(P: np.ndarray):
    """Decompose camera projection matrix P (3x4) to intrinsics K and pose (c2w).
    Borrowed pattern from IDR; returns 4x4 intrinsics (top-left 3x3 is K) and 4x4 c2w pose.
    """
    import cv2 as cv
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K.astype(np.float32)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose().astype(np.float32)
    pose[:3, 3] = (t[:3] / t[3])[:, 0].astype(np.float32)
    return intrinsics, pose  # c2w


def carve_visual_hull(data_dir: str, n_views: int = 8, res: int = 64, erode: int = 1,
                      bbox_min=(-1.0, -1.0, -1.0), bbox_max=(1.0, 1.0, 1.0)):
    # Load camera mats and masks
    cam_path = os.path.join(data_dir, "cameras_sphere.npz")
    cam_dict = np.load(cam_path)

    # Gather mask files and image size
    mask_dir = os.path.join(data_dir, "mask")
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
    if len(mask_files) == 0:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    # Use the indices inferred from filenames to fetch the right world/scale mats
    img_indices = [int(os.path.splitext(os.path.basename(p))[0]) for p in mask_files]
    # Select first n_views consistently
    sel = list(range(min(n_views, len(img_indices))))

    # Read masks and camera parameters
    masks = []
    Ks = []
    c2ws = []
    H, W = None, None
    for k in sel:
        idx = img_indices[k]
        m = cv.imread(mask_files[k], cv.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {mask_files[k]}")
        if erode > 0:
            m = cv.erode(m, np.ones((erode, erode), np.uint8))
        masks.append((m > 127).astype(np.uint8))
        if H is None:
            H, W = m.shape

        P = (cam_dict[f'world_mat_{idx}'] @ cam_dict[f'scale_mat_{idx}']).astype(np.float32)
        P = P[:3, :4]
        K4, c2w = load_K_Rt_from_P(P)
        Ks.append(K4[:3, :3])
        c2ws.append(c2w)

    Ks = np.stack(Ks, axis=0)
    c2ws = np.stack(c2ws, axis=0)
    w2cs = np.linalg.inv(c2ws)

    # Build voxel grid in world coordinates
    bmin = np.array(bbox_min, dtype=np.float32)
    bmax = np.array(bbox_max, dtype=np.float32)
    grid = np.stack(np.meshgrid(
        np.linspace(bmin[0], bmax[0], res, endpoint=False) + (bmax[0]-bmin[0])/(2*res),
        np.linspace(bmin[1], bmax[1], res, endpoint=False) + (bmax[1]-bmin[1])/(2*res),
        np.linspace(bmin[2], bmax[2], res, endpoint=False) + (bmax[2]-bmin[2])/(2*res),
        indexing='ij'
    ), axis=-1)  # (res,res,res,3)
    occ = np.ones((res, res, res), dtype=np.uint8)

    pts = grid.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1).T  # (4, N)

    # Project into each view and intersect masks
    for v in range(len(sel)):
        K = Ks[v]
        w2c = w2cs[v]
        Xc = (w2c @ pts_h)  # (4,N)
        Z = Xc[2, :]
        valid_z = Z > 1e-6
        x = (K[0, 0] * (Xc[0, :] / Z) + K[0, 2])
        y = (K[1, 1] * (Xc[1, :] / Z) + K[1, 2])
        u = np.round(x).astype(np.int32)
        vpix = np.round(y).astype(np.int32)
        inside = (u >= 0) & (u < W) & (vpix >= 0) & (vpix < H) & valid_z
        maskv = np.zeros_like(inside, dtype=np.uint8)
        mask_img = masks[v]
        maskv[inside] = mask_img[vpix[inside], u[inside]] > 0
        occ = occ & maskv.reshape(res, res, res)

    # Marching cubes
    occ_f = occ.astype(np.float32)
    if occ_f.max() <= 0.0:
        raise RuntimeError("Visual hull is empty at this resolution/views. Try increasing --views or --res.")

    verts, faces, normals, _ = measure.marching_cubes(occ_f, level=0.5)

    # Map voxel coordinates to world
    voxel_size = (bmax - bmin) / res
    verts_world = bmin + (verts + 0.5) * voxel_size

    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_world.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.compute_vertex_normals()

    return mesh, (H, W), (Ks, c2ws)


def save_mesh_and_preview(mesh: o3d.geometry.TriangleMesh, out_ply: str, out_png: str):
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_triangle_mesh(out_ply, mesh)

    # Matplotlib preview
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    tri = Poly3DCollection(v[f], alpha=0.9)
    tri.set_edgecolor('k')
    tri.set_linewidth(0.1)
    tri.set_facecolor((0.7, 0.7, 0.85))
    ax.add_collection3d(tri)
    ax.auto_scale_xyz(v[:, 0], v[:, 1], v[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(os.path.basename(out_ply))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/diligent_mv_normals/bear')
    parser.add_argument('--views', type=int, default=8)
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--erode', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='exp/visual_hull')
    parser.add_argument('--name', type=str, default='bear_hull')
    args = parser.parse_args()

    mesh, _, _ = carve_visual_hull(args.data_dir, n_views=args.views, res=args.res, erode=args.erode)
    out_ply = os.path.join(args.outdir, f"{args.name}_v{args.views}_r{args.res}.ply")
    out_png = os.path.join(args.outdir, f"{args.name}_v{args.views}_r{args.res}.png")
    save_mesh_and_preview(mesh, out_ply, out_png)
    print(f"✓ Saved: {out_ply}")
    print(f"✓ Saved: {out_png}")


if __name__ == '__main__':
    main()
