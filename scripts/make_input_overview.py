#!/usr/bin/env python3
"""
Create a visual overview showing input images, masks, and reconstruction results for Review 2 demo.
"""
import argparse
import os
import glob
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data/diligent_mv_normals/bear')
    ap.add_argument('--n_views', type=int, default=4, help='Number of views to show')
    ap.add_argument('--out', type=str, default='exp/for_guide/input_overview.png')
    args = ap.parse_args()

    # Find RGB images (use integration/rgb if available, else try other folders)
    rgb_dir = os.path.join(args.data_dir, 'integration', 'rgb')
    if not os.path.exists(rgb_dir):
        rgb_dir = os.path.join(args.data_dir, 'images')  # fallback
    
    if os.path.exists(rgb_dir):
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')) + 
                          glob.glob(os.path.join(rgb_dir, '*.jpg')))[:args.n_views]
    else:
        rgb_files = []
    
    mask_files = sorted(glob.glob(os.path.join(args.data_dir, 'mask', '*.png')))[:args.n_views]
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Create figure with 2 rows: RGB images (if available) and masks
    if len(rgb_files) > 0:
        fig, axes = plt.subplots(2, args.n_views, figsize=(12, 5))
        fig.suptitle('Input: Multi-view images and silhouette masks', fontsize=14)
        
        for i in range(args.n_views):
            # RGB
            if i < len(rgb_files):
                img = cv.imread(rgb_files[i])
                if img is not None:
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'View {i}')
                    axes[0, i].axis('off')
            
            # Mask
            if i < len(mask_files):
                mask = cv.imread(mask_files[i], cv.IMREAD_GRAYSCALE)
                if mask is not None:
                    axes[1, i].imshow(mask, cmap='gray')
                    axes[1, i].set_title(f'Mask {i}')
                    axes[1, i].axis('off')
    else:
        # Only masks available
        fig, axes = plt.subplots(1, args.n_views, figsize=(12, 3))
        fig.suptitle('Input: Silhouette masks from multi-view capture', fontsize=14)
        
        for i in range(min(args.n_views, len(mask_files))):
            mask = cv.imread(mask_files[i], cv.IMREAD_GRAYSCALE)
            if mask is not None:
                if args.n_views > 1:
                    axes[i].imshow(mask, cmap='gray')
                    axes[i].set_title(f'View {i}')
                    axes[i].axis('off')
                else:
                    axes.imshow(mask, cmap='gray')
                    axes.set_title(f'View {i}')
                    axes.axis('off')
    
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {args.out}')
    
    # Also create a simple pipeline diagram text file
    pipeline_txt = os.path.join(os.path.dirname(args.out), 'pipeline_summary.txt')
    with open(pipeline_txt, 'w') as f:
        f.write("PMNI-inspired Reconstruction Pipeline (CPU-first)\n")
        f.write("=" * 60 + "\n\n")
        f.write("INPUTS:\n")
        f.write(f"  - Dataset: {os.path.basename(args.data_dir)}\n")
        f.write(f"  - Views: {len(mask_files)} (used first 8)\n")
        f.write(f"  - Data: Multi-view images + binary masks + calibrated cameras\n")
        f.write(f"  - Camera file: cameras_sphere.npz (world_mat, scale_mat per view)\n\n")
        f.write("STAGE 1: Visual Hull (Silhouette-based)\n")
        f.write("  - Method: Voxel carving from multi-view silhouettes\n")
        f.write("  - Resolution: 64³ voxel grid\n")
        f.write("  - Algorithm: Project each voxel → keep if inside ALL masks\n")
        f.write("  - Output: Coarse watertight mesh (PLY)\n")
        f.write("  - Why textureless-friendly: Uses only shape silhouettes\n\n")
        f.write("STAGE 2: Clean Point Cloud\n")
        f.write("  - Method: Poisson disk sampling on mesh surface\n")
        f.write("  - Points: 250,000 uniformly sampled\n")
        f.write("  - Output: Dense point cloud (PLY)\n\n")
        f.write("NEXT (PMNI-lite refinement):\n")
        f.write("  - Fit SDF MLP with normal-consistency loss\n")
        f.write("  - Losses: surface + Eikonal + normal alignment\n")
        f.write("  - Extract refined mesh via marching cubes\n")
        f.write("  - CPU-friendly variant of PMNI's approach\n")
    
    print(f'✓ Saved: {pipeline_txt}')


if __name__ == '__main__':
    main()
