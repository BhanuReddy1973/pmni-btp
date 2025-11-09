# PMNI-lite: CPU-Friendly Neural Surface Reconstruction

A lightweight implementation of PMNI (Photometric Normal Integration) designed to run on CPU without CUDA-only dependencies (nerfacc, tiny-cuda-nn).

## Overview

PMNI-lite implements a neural implicit surface representation using:
- **SDF Network**: MLP with geometric initialization and optional positional encoding
- **Normal Supervision**: Strong normal consistency loss for textureless scenes
- **Sphere Marching Renderer**: CPU-based ray marching without nerfacc
- **Multi-view Data**: Depth, normals, and masks from calibrated views

## Architecture

```
pmni_lite/
├── sdf_network.py      # SDF MLP with geometric initialization
├── dataset.py          # Multi-view dataset loader
├── renderer.py         # CPU sphere-marching renderer
├── losses.py           # Surface, eikonal, normal, mask, depth losses
├── trainer.py          # Training loop and mesh extraction
└── train.py            # Main entry point
```

## Quick Start

### Training on DiLiGenT-MV Bear

```bash
cd /home/bhanu/pmni/PMNI
source scripts/activate_pmni.sh

python3 pmni_lite/train.py \
  --data_dir data/diligent_mv_normals/bear \
  --views 8 \
  --n_iters 5000 \
  --batch_size 512 \
  --exp_dir exp/pmni_lite/bear_run1
```

### Key Parameters

- `--views`: Number of views to use (default: 8)
- `--hidden_dim`: MLP hidden dimension (default: 256)
- `--n_layers`: Number of layers (default: 8)
- `--multires`: Positional encoding frequencies, 0 to disable (default: 6)
- `--normal_weight`: Weight for normal consistency loss (default: 1.0)
- `--eikonal_weight`: Weight for Eikonal regularization (default: 0.1)

### Output

```
exp/pmni_lite/bear_run1/
├── checkpoints/
│   ├── ckpt_001000.pth
│   ├── ckpt_002000.pth
│   └── final.pth
└── meshes/
    ├── mesh_000500.ply
    ├── mesh_001000.ply
    └── final.ply
```

## Differences from Full PMNI

| Feature | Full PMNI | PMNI-lite |
|---------|-----------|-----------|
| Device | CUDA (required) | CPU (CUDA optional) |
| Ray marching | nerfacc OccupancyGrid | Sphere marching |
| Encoding | tiny-cuda-nn | PyTorch sin/cos |
| Speed | Fast (GPU) | Slower (CPU) |
| Dependencies | CUDA, nerfacc, tcnn | PyTorch, NumPy, Open3D |

## Review 2 Deliverables

For Review 2, we demonstrate:

1. **Complete implementation** of PMNI-lite with all core components
2. **Trained model** on bear dataset (8 views, 5000 iters)
3. **Extracted mesh** showing improved geometry over visual hull baseline
4. **Comparison**: Visual hull (silhouette-only) vs. PMNI-lite (silhouette + normals)
5. **Provenance**: Full reproducibility with config logging

## Next Steps (Review 3)

- Add quantitative evaluation metrics (Chamfer distance, normal MAE)
- Implement COLMAP integration for arbitrary photo sets
- Add view synthesis and novel normal prediction
- Optimize rendering speed with better acceleration structures

## Next Steps (Review 4)

- Streamlit web interface for end-to-end demo
- GPU acceleration path (optional nerfacc integration)
- Multi-object support and batch processing
