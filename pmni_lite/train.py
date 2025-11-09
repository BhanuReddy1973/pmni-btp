"""
Main training script for PMNI-lite.
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmni_lite.sdf_network import SDFNetwork
from pmni_lite.dataset import MultiViewDataset
from pmni_lite.renderer import SphereMarchingRenderer
from pmni_lite.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='PMNI-lite training')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='data/diligent_mv_normals/bear',
                       help='Path to dataset directory')
    parser.add_argument('--views', type=int, default=8,
                       help='Number of views to use')
    
    # Network
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=8,
                       help='Number of layers')
    parser.add_argument('--multires', type=int, default=6,
                       help='Positional encoding frequencies (0 to disable)')
    
    # Rendering
    parser.add_argument('--n_samples', type=int, default=64,
                       help='Number of samples per ray')
    parser.add_argument('--max_steps', type=int, default=128,
                       help='Max sphere marching steps')
    
    # Training
    parser.add_argument('--n_iters', type=int, default=5000,
                       help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size (number of rays)')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    
    # Loss weights
    parser.add_argument('--sdf_weight', type=float, default=1.0)
    parser.add_argument('--eikonal_weight', type=float, default=0.1)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument('--mask_weight', type=float, default=0.1)
    parser.add_argument('--depth_weight', type=float, default=0.5)
    
    # Logging
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--val_every', type=int, default=500)
    parser.add_argument('--ckpt_every', type=int, default=1000)
    
    # Output
    parser.add_argument('--exp_dir', type=str, default='exp/pmni_lite/bear_run1',
                       help='Experiment directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print('='*60)
    print('PMNI-lite Training')
    print('='*60)
    print(f'Data: {args.data_dir}')
    print(f'Views: {args.views}')
    print(f'Network: {args.n_layers} layers x {args.hidden_dim} hidden, multires={args.multires}')
    print(f'Training: {args.n_iters} iters, batch={args.batch_size}, lr={args.lr}')
    print(f'Device: {args.device}')
    print(f'Output: {args.exp_dir}')
    print('='*60)
    
    # Load dataset
    print('\nLoading dataset...')
    dataset = MultiViewDataset(
        data_dir=args.data_dir,
        views=args.views,
        device=args.device
    )
    
    # Create network
    print('\nCreating SDF network...')
    sdf_network = SDFNetwork(
        d_hidden=args.hidden_dim,
        n_layers=args.n_layers,
        multires=args.multires,
        geometric_init=True,
        weight_norm=True
    )
    
    # Create renderer
    print('Creating renderer...')
    renderer = SphereMarchingRenderer(
        sdf_network=sdf_network,
        n_samples=args.n_samples,
        max_steps=args.max_steps
    )
    
    # Create trainer
    print('Creating trainer...')
    trainer = Trainer(
        sdf_network=sdf_network,
        renderer=renderer,
        dataset=dataset,
        exp_dir=args.exp_dir,
        device=args.device,
        lr=args.lr,
        sdf_weight=args.sdf_weight,
        eikonal_weight=args.eikonal_weight,
        normal_weight=args.normal_weight,
        mask_weight=args.mask_weight,
        depth_weight=args.depth_weight
    )
    
    # Train
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60 + '\n')
    
    trainer.train(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        log_every=args.log_every,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every
    )
    
    print('\n' + '='*60)
    print('Training complete!')
    print(f'Results saved to: {args.exp_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
