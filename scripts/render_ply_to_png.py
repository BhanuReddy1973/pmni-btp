#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

def load_points_from_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        pts = np.asarray(pc.points)
        return pts
    # fallback: simple ASCII PLY reader (vertex only)
    pts = []
    with open(path, 'r') as f:
        header = True
        for line in f:
            if header:
                if line.strip() == 'end_header':
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = map(float, parts[:3])
                    pts.append((x, y, z))
                except ValueError:
                    continue
    return np.array(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True, help='Path to PLY file')
    ap.add_argument('--out', required=False, default=None, help='Output PNG path')
    ap.add_argument('--step', type=int, default=200, help='Subsample step for plotting')
    args = ap.parse_args()

    assert os.path.exists(args.ply), f"PLY not found: {args.ply}"
    pts = load_points_from_ply(args.ply)
    if pts.size == 0:
        raise RuntimeError('No points loaded from PLY')

    if args.out is None:
        base, _ = os.path.splitext(args.ply)
        args.out = base + '_preview.png'

    # compute stats
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    print(f'Loaded {pts.shape[0]} points')
    print(f'Bounds: min {mins}, max {maxs}')

    idx = np.arange(0, pts.shape[0], max(1, pts.shape[0] // 50000))  # cap ~50k points
    sub = pts[idx]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sub[:,0], sub[:,1], sub[:,2], s=0.3, c='k', alpha=0.5)
    ax.set_title(os.path.basename(args.ply))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f'Saved preview: {args.out}')

if __name__ == '__main__':
    main()
