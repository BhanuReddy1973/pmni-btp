# Simplified PMNI - Lightweight Neural SDF with Normal Supervision

A computationally efficient reimplementation of PMNI for resource-constrained environments.

## Key Simplifications

1. **No Complex Dependencies**:
   - ❌ No `tiny-cuda-nn` (plain PyTorch MLP)
   - ❌ No `nerfacc` (simple volume rendering)
   - ✅ Just PyTorch + NumPy + OpenCV

2. **Lightweight Architecture**:
   - Simple 4-layer MLP (configurable)
   - Uniform ray sampling (no occupancy grid)
   - Direct autodiff for gradients (no finite differences)

3. **Memory Efficient**:
   - Dataset stays on CPU
   - Batch processing of rays
   - No large intermediate buffers

## Usage

### Quick Start

```bash
# Activate PMNI environment
cd /home/bhanu/pmni/PMNI
source scripts/activate_pmni.sh

# Run simplified training
cd simple_pmni
python train.py --obj_name bear --iterations 500 --batch_size 512 --hidden_dim 64
```

### Configuration Options

```bash
python train.py \
    --data_dir ./data/diligent_mv_normals \
    --obj_name bear \
    --output_dir ./simple_output \
    --iterations 500 \
    --batch_size 512 \
    --lr 1e-3 \
    --hidden_dim 64 \
    --num_layers 4 \
    --mesh_res 128 \
    --device cuda  # or 'cpu' for CPU-only training
```

### For Very Limited GPU Memory

```bash
# Ultra-minimal config
python train.py \
    --batch_size 256 \
    --hidden_dim 32 \
    --num_layers 3 \
    --iterations 300
```

## Architecture

### Model (`model.py`)
- `SimpleSDF`: Plain MLP for signed distance field
- `VarianceNetwork`: Learnable surface thickness
- `SimplePMNI`: Combined model

### Renderer (`renderer.py`)
- `generate_rays()`: Camera ray generation
- `simple_volume_rendering()`: Uniform sampling volume rendering
- `render_normals()`: Render surface normals

### Dataset (`dataset.py`)
- `DiLiGentDataset`: Simplified loader for DiLiGenT-MV data

### Trainer (`train.py`)
- Training loop with normal + mask + eikonal losses
- Checkpoint saving
- Mesh extraction using marching cubes

## Losses

1. **Normal Loss**: Cosine similarity + L1 between predicted and GT normals
2. **Mask Loss**: Binary cross-entropy on rendered opacity vs silhouette
3. **Eikonal Loss**: Regularization for unit gradient norm (SDF property)

## Expected Performance

- **Memory**: ~1-2 GB GPU (with batch_size=512, hidden_dim=64)
- **Speed**: ~5 min for 500 iterations on GPU
- **Quality**: Reasonable reconstruction for Review 2 demo

## Output

- `output/ckpt_XXXXX.pth`: Model checkpoints every 100 iterations
- `output/final_mesh.ply`: Extracted mesh (view with MeshLab/CloudCompare)

## Testing Components

```bash
# Test model
python model.py

# Test renderer  
python renderer.py

# Test dataset
python dataset.py
```

## Differences from Original PMNI

| Feature | Original PMNI | Simplified |
|---------|---------------|------------|
| Encoding | tiny-cuda-nn hash grid | None (plain MLP) |
| Ray sampling | nerfacc occupancy grid | Uniform sampling |
| Pose optimization | LearnPose network | Fixed (GT poses) |
| Scale learning | ScaleNet | Fixed |
| Dependencies | 10+ libraries | 3 libraries |
| GPU memory | ~8-10 GB | ~1-2 GB |
| Training speed | Medium | Faster (simpler) |

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 256, 128, 64)
- Reduce `--hidden_dim` (try 32, 16)
- Use `--device cpu` for CPU-only training (slower)

### NaN Losses
- Check dataset loading (run `python dataset.py`)
- Reduce learning rate: `--lr 5e-4`
- Check GPU memory isn't corrupted (restart kernel)

### No Mesh Output
- Check if checkpoints are saved in `output/`
- Run mesh extraction manually after training
- Lower `--mesh_res` if OOM during extraction

## For Your Guide

**What to present**:
1. This simplified implementation (show the code structure)
2. Training logs showing stable losses
3. Extracted mesh visualization
4. Comparison with original PMNI complexity

**Key points**:
- Built from scratch to understand the method
- Removed computational bottlenecks
- Maintains core idea (neural SDF + normal supervision)
- Can run on limited resources
