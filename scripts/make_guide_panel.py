#!/usr/bin/env python3
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc_png', type=str, required=True)
    ap.add_argument('--vh_png', type=str, required=True)
    ap.add_argument('--out', type=str, default='exp/for_guide/baseline_panel.png')
    ap.add_argument('--title', type=str, default='Textureless Scene Baselines')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img_pc = mpimg.imread(args.pc_png)
    img_vh = mpimg.imread(args.vh_png)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(args.title)
    axes[0].imshow(img_pc)
    axes[0].axis('off')
    axes[0].set_title('Depth-fused point cloud (CPU)')
    axes[1].imshow(img_vh)
    axes[1].axis('off')
    axes[1].set_title('Visual hull mesh (CPU)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(args.out, dpi=160)
    print(f"✓ Saved: {args.out}")


if __name__ == '__main__':
    main()
