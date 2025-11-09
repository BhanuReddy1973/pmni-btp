#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
import numpy as np
import glob


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except Exception:
        return None


def visual_hull_provenance(data_dir: str, out_mesh: str, views: int, res: int, erode: int):
    cam_file = os.path.join(data_dir, 'cameras_sphere.npz')
    cam = np.load(cam_file)
    mask_files = sorted(glob.glob(os.path.join(data_dir, 'mask', '*.png')))
    idxs = [int(os.path.splitext(os.path.basename(p))[0]) for p in mask_files]
    sel = idxs[:views]
    masks_used = [mask_files[i] for i in range(min(views, len(mask_files)))]
    return {
        "artifact": os.path.abspath(out_mesh),
        "algorithm": "visual_hull",
        "script": "scripts/visual_hull.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit(),
        "data_dir": os.path.abspath(data_dir),
        "cameras_file": os.path.abspath(cam_file),
        "view_indices": sel,
        "masks": [os.path.abspath(p) for p in masks_used],
        "params": {"views": views, "res": res, "erode": erode}
    }


def mesh_sampling_provenance(src_mesh: str, out_ply: str, n_points: int):
    return {
        "artifact": os.path.abspath(out_ply),
        "algorithm": "mesh_poisson_sampling",
        "script": "scripts/mesh_to_pointcloud.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit(),
        "source_mesh": os.path.abspath(src_mesh),
        "params": {"n_points": n_points, "method": "poisson_disk"}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vh_data_dir', type=str, default='data/diligent_mv_normals/bear')
    ap.add_argument('--vh_out_mesh', type=str, default='exp/visual_hull/bear_hull_v8_r64.ply')
    ap.add_argument('--vh_views', type=int, default=8)
    ap.add_argument('--vh_res', type=int, default=64)
    ap.add_argument('--vh_erode', type=int, default=1)

    ap.add_argument('--pc_src_mesh', type=str, default='exp/visual_hull/bear_hull_v8_r64.ply')
    ap.add_argument('--pc_out_ply', type=str, default='exp/quick_outputs/bear_pc_from_mesh.ply')
    ap.add_argument('--pc_n_points', type=int, default=250000)
    args = ap.parse_args()

    # Visual hull provenance
    vh = visual_hull_provenance(args.vh_data_dir, args.vh_out_mesh, args.vh_views, args.vh_res, args.vh_erode)
    vh_path = os.path.splitext(args.vh_out_mesh)[0] + '.provenance.json'
    os.makedirs(os.path.dirname(vh_path), exist_ok=True)
    with open(vh_path, 'w') as f:
        json.dump(vh, f, indent=2)

    # Mesh-sampled point cloud provenance
    pc = mesh_sampling_provenance(args.pc_src_mesh, args.pc_out_ply, args.pc_n_points)
    pc_path = os.path.splitext(args.pc_out_ply)[0] + '.provenance.json'
    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
    with open(pc_path, 'w') as f:
        json.dump(pc, f, indent=2)

    print('✓ Wrote:', vh_path)
    print('✓ Wrote:', pc_path)


if __name__ == '__main__':
    main()
