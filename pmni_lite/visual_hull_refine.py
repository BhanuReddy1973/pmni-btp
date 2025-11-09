"""
Compute a visual hull via voxel carving from masks, save hull mesh, and
optionally pretrain a small SDF network to the hull (quick PMNI-style init).

Usage (from repo root):
PYTHONPATH=/home/bhanu/pmni python3 -m PMNI.pmni_lite.visual_hull_refine \
    --data_dir PMNI/data/diligent_mv_normals/bear --exp_dir exp/pmni_lite/hull_run --res 128

"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path

from .dataset import MultiViewDataset

def carve_voxels(dataset, bbox_min, bbox_max, res=128, chunk=100000):
    xs = np.linspace(bbox_min[0], bbox_max[0], res)
    ys = np.linspace(bbox_min[1], bbox_max[1], res)
    zs = np.linspace(bbox_min[2], bbox_max[2], res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).astype(np.float32)
    pts = grid.reshape(-1, 3)
    occ = np.ones(pts.shape[0], dtype=bool)

    # Precompute intrinsics and pose inverses
    intr = dataset.intrinsics_all.numpy()[:, :3, :3]
    poses = dataset.pose_all.numpy()
    pose_inv = np.linalg.inv(poses)

    H, W = dataset.H, dataset.W

    for vid in range(dataset.n_images):
        K = intr[vid]
        Pinv = pose_inv[vid]

        # Process in chunks to limit memory
        for i in range(0, pts.shape[0], chunk):
            p_chunk = pts[i:i+chunk]
            # Transform to camera coords: X_cam = pose_inv @ [X,1]
            hom = np.concatenate([p_chunk, np.ones((p_chunk.shape[0], 1), dtype=np.float32)], axis=1)
            Xcam = (Pinv @ hom.T).T[:, :3]

            z = Xcam[:, 2]
            visible = z > 1e-4
            if not visible.any():
                continue

            proj = (K @ Xcam[visible].T).T
            u = (proj[:, 0] / proj[:, 2]).astype(np.int32)
            v = (proj[:, 1] / proj[:, 2]).astype(np.int32)

            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idxs = np.nonzero(visible)[0][inside]
            u_in = u[inside]
            v_in = v[inside]

            if len(idxs) == 0:
                continue

            masks = dataset.masks[vid].numpy()
            # If any view sees the voxel as background (mask==0), carve it
            mask_vals = masks[v_in, u_in]
            carved = mask_vals < 0.5
            occ[i + idxs[carved]] = False

    vol = occ.reshape((res, res, res))
    return vol, (xs, ys, zs)


def extract_mesh_from_volume(vol, bbox_min, bbox_max, res, out_path):
    try:
        from skimage import measure
    except ModuleNotFoundError:
        print('skimage not installed; cannot extract hull mesh')
        return None

    try:
        verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=0.5)
    except Exception as e:
        print('marching_cubes failed:', e)
        return None

    verts_world = np.array(bbox_min) + (np.array(bbox_max) - np.array(bbox_min)) * (verts / (res - 1.0))

    try:
        import open3d as o3d
    except ModuleNotFoundError:
        print('open3d not installed; cannot write mesh')
        return None

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_world.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print('Wrote hull mesh to', out_path)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--exp_dir', default='exp/pmni_lite/hull_run')
    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--bbox_min', type=float, nargs=3, default=[-0.6, -0.6, -0.1])
    parser.add_argument('--bbox_max', type=float, nargs=3, default=[0.6, 0.6, 0.8])
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    mesh_out = Path(args.exp_dir) / 'visual_hull.ply'

    print('Loading dataset...')
    # Auto-detect normal folder name (DiLiGenT variants)
    normal_dir = 'normal'
    for cand in ('normal_camera_space_GT', 'normal_world_space_GT', 'normal'):
        if os.path.isdir(os.path.join(args.data_dir, cand)):
            normal_dir = cand
            break
    print(f'Using normal_dir={normal_dir}')
    ds = MultiViewDataset(args.data_dir, normal_dir=normal_dir, views=8, device='cpu')

    print('Carving voxels... (this may take a while)')
    vol, grids = carve_voxels(ds, args.bbox_min, args.bbox_max, res=args.res)

    print('Extracting mesh from carved volume...')
    mesh = extract_mesh_from_volume(vol, args.bbox_min, args.bbox_max, args.res, mesh_out)

    if mesh is None:
        print('Mesh extraction failed')
        return

    # Optionally: quick SDF pretrain to hull (small network, few iters)
    try:
        from .sdf_network import SDFNetwork
        import open3d as o3d
    except Exception:
        print('Skipping SDF pretrain because dependencies missing')
        return

    print('Building small SDF network and pretraining to hull (fast)...')
    device = torch.device('cpu')
    net = SDFNetwork(d_hidden=128, n_layers=6, multires=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Build KDTree for hull vertices
    hull_verts = np.asarray(mesh.vertices)
    # Build KDTree using Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hull_verts)
    kdt = o3d.geometry.KDTreeFlann(pcd)

    # Sample points in bbox and train to hull distance
    xs = np.linspace(args.bbox_min[0], args.bbox_max[0], args.res)
    ys = np.linspace(args.bbox_min[1], args.bbox_max[1], args.res)
    zs = np.linspace(args.bbox_min[2], args.bbox_max[2], args.res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).astype(np.float32)
    pts = grid.reshape(-1, 3)

    # Determine occupancy (inside/outside) from carved volume
    occ = vol.reshape(-1)

    # Quick training loop: sample random points and set target sdf = signed distance to hull
    n_iters = 200
    batch = 16384
    for it in range(n_iters):
        ids = np.random.randint(0, pts.shape[0], size=min(batch, pts.shape[0]))
        ps = pts[ids]
        # Query KDTree for nearest hull vertex. If KDTree search raises, fall back to
        # numpy brute-force distance (vectorized) for the batch.
        try:
            ds_vals = []
            for p in ps:
                [k, idx, d2] = kdt.search_knn_vector_3d(p.tolist(), 1)
                ds_vals.append(np.sqrt(d2[0]))
            ds_vals = np.array(ds_vals, dtype=np.float32)
        except Exception:
            # Fallback: vectorized distance to hull vertices
            dists = np.sqrt(((ps[:, None, :] - hull_verts[None, :, :]) ** 2).sum(-1))
            ds_vals = dists.min(axis=1).astype(np.float32)
        signs = np.where(occ[ids], -1.0, 1.0).astype(np.float32)
        sdf_t = ds_vals * signs

        pts_t = torch.from_numpy(ps).float().to(device)
        sdf_t_t = torch.from_numpy(sdf_t).float().to(device)

        pred = net(pts_t).squeeze(-1)
        loss = torch.nn.functional.l1_loss(pred, sdf_t_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            print(f'Pretrain iter {it+1}/{n_iters} loss={loss.item():.4f}')

    # Save the pretrained network
    ckpt_path = Path(args.exp_dir) / 'sdf_pretrain_to_hull.pth'
    torch.save(net.state_dict(), str(ckpt_path))
    print('Saved pretrain checkpoint to', ckpt_path)


if __name__ == '__main__':
    main()
