#!/usr/bin/env python3
"""
Sample a clean point cloud from a mesh (e.g., visual hull output) for prettier visualization.
"""
import argparse
import os
import numpy as np

import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=str, required=True)
    ap.add_argument('--out_ply', type=str, required=True)
    ap.add_argument('--out_png', type=str, required=True)
    ap.add_argument('--n_points', type=int, default=200000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    pc = mesh.sample_points_poisson_disk(number_of_points=args.n_points)
    o3d.io.write_point_cloud(args.out_ply, pc)

    pts = np.asarray(pc.points)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, pts.shape[0] // 200000)
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=0.2, c='black')
    ax.set_title(os.path.basename(args.out_ply))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    print('✓ Saved:', args.out_ply)
    print('✓ Saved:', args.out_png)


if __name__ == '__main__':
    main()
