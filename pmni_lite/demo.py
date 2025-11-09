"""
Tiny demo runner for PMNI-lite using the synthetic sphere dataset.

This is designed for quick runs on CPU for review demos. It runs a very small
number of iterations and saves a mesh to the experiment directory.
"""
import argparse
import os

import torch

from .sdf_network import SDFNetwork
from .renderer import SphereMarchingRenderer
from .trainer import Trainer
from .synthetic_dataset import MultiViewSyntheticDataset


def run_demo(exp_dir='exp/pmni_lite/demo', n_iters=50, batch_size=256, device='cpu', data_dir=''):
    os.makedirs(exp_dir, exist_ok=True)

    # Choose dataset: real multi-view dataset if provided, else synthetic
    if data_dir is not None and data_dir != '' and os.path.exists(data_dir):
        # Some systems may not have EXR reader dependencies installed (pyexr).
        # Try to import pyexr and fall back to synthetic dataset if missing.
        try:
            import pyexr  # noqa: F401
            from .dataset import MultiViewDataset
            # Auto-detect common normal folder names used in datasets (DiLiGenT variants)
            normal_dir = 'normal'
            for cand in ('normal_camera_space_GT', 'normal_world_space_GT', 'normal'):
                if os.path.isdir(os.path.join(data_dir, cand)):
                    normal_dir = cand
                    break

            print(f'Using real dataset at: {data_dir} (normal_dir={normal_dir})')
            dataset = MultiViewDataset(data_dir=data_dir, normal_dir=normal_dir, views=8, device=device)
        except Exception:
            print("pyexr not available or failed to import — falling back to synthetic dataset for demo")
            dataset = MultiViewSyntheticDataset(n_views=8, H=64, W=64, device=device)
    else:
        if data_dir:
            print(f"Warning: data_dir '{data_dir}' not found, falling back to synthetic dataset")
        else:
            print('No data_dir provided, using synthetic dataset')
        dataset = MultiViewSyntheticDataset(n_views=8, H=64, W=64, device=device)

    print('Creating network (small)...')
    sdf_network = SDFNetwork(d_hidden=128, n_layers=6, multires=4, predict_normals=False)

    print('Creating renderer...')
    renderer = SphereMarchingRenderer(sdf_network=sdf_network, n_samples=32, max_steps=64)

    print('Creating trainer...')
    trainer = Trainer(
        sdf_network=sdf_network,
        renderer=renderer,
        dataset=dataset,
        exp_dir=exp_dir,
        device=device,
        lr=5e-4,
        sdf_weight=1.0,
        eikonal_weight=0.1,
        normal_weight=1.0,
        mask_weight=0.1,
        depth_weight=0.5,
    )

    print('\nStarting short demo training...')
    # For quick demo runs on CPU we skip intermediate mesh extraction/checkpointing
    trainer.train(n_iters=n_iters, batch_size=batch_size, log_every=10, val_every=0, ckpt_every=0)

    # Try adaptive mesh extraction and save result (will gracefully skip if mesh libs missing)
    print('Attempting adaptive mesh extraction...')
    trainer.validate_and_save_mesh('adaptive_final.ply')

    print('Demo finished. Mesh (if generated) is in', os.path.join(exp_dir, 'meshes'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='exp/pmni_lite/demo')
    parser.add_argument('--n_iters', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data_dir', type=str, default='')
    args = parser.parse_args()

    run_demo(exp_dir=args.exp_dir, n_iters=args.n_iters, batch_size=args.batch_size, device=args.device, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
