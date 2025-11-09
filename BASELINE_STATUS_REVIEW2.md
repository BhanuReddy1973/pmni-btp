# PMNI Baseline Implementation - Review 2 Status

**Project:** Challenges of 3D Reconstruction in Textureless Scenes: A Hybrid Approach  
**Date:** November 5, 2025  
**Student:** Bhanu

---

## Executive Summary

**Attempted to implement PMNI baseline for textureless 3D reconstruction.**  
**Status: Implementation complete but BLOCKED by GPU memory constraints.**

---

## What Was Done

### 1. Code Integration ✅
- Integrated PMNI (Photometric Multi-view Neural Implicit) codebase
- Set up DiLiGenT-MV dataset (bear object, 20 views with GT normals)
- Configured training pipeline with:
  - Normal supervision loss
  - Mask loss (silhouette)
  - Eikonal regularization
  - Depth consistency

### 2. Optimization Attempts ✅
Created progressive configurations to reduce memory:
- **Minimal config**: 5 views only, 32 hidden units, 2 layers, no hash encoding
- **Dataset optimization**: Store data on CPU, transfer batches to GPU
- **Disabled features**: No pose learning (lr=0), no scale learning, no weight normalization
- **Tiny occupancy grid**: 32³ resolution (vs 128³ default)
- **Small batch**: 64 rays/batch (vs 4096 default)

### 3. Code Fixes ✅
- Made encoding parameter optional (works without tiny-cuda-nn)
- Fixed dataset to support CPU storage
- Created minimal baseline configuration

---

## Current Blocking Issue

### GPU Memory Status
```
MIG Device 0: 19,827 MiB / 20,096 MiB used = 269 MiB free
```

**Only 269 MB free - insufficient for ANY model initialization.**

### Error Sequence
1. ✅ Dataset loads successfully (on CPU)
2. ✅ SDF network creates successfully (plain MLP, no encoding)
3. ❌ **FAILS** when creating Occupancy Grid on GPU:
   ```
   RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED
   CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
   ```

**Root cause:** GPU memory is fragmented/exhausted. Even minimal allocation (OccupancyGrid initialization) triggers CUDA error.

---

## Technical Details

### Configuration Created
**File:** `config/minimal_baseline.conf`
- **Views:** 5 (excluded 15 views to reduce memory)
- **Iterations:** 50 (smoke test)
- **Batch size:** 64 rays
- **Network:** 2-layer MLP, 32 hidden units, NO encoding
- **Occupancy grid:** 32³ resolution
- **Pose/Scale learning:** Disabled (lr=0)
- **Mesh resolution:** 64³ (very low)

### Dataset
- **Object:** DiLiGenT-MV bear
- **Data:** 512×612 normal maps, depth maps, masks
- **Storage:** CPU (to avoid GPU OOM)
- **Cameras:** GT calibration from dataset

### Training Pipeline
- Ray marching with occupancy grid acceleration
- Normal loss (L2)
- Mask loss (silhouette)  
- Eikonal regularization
- No pose/scale optimization

---

## Deliverables Available

### Code ✅
1. `/home/bhanu/pmni/PMNI/exp_runner.py` - Main training script (modified for optional encoding)
2. `/home/bhanu/pmni/PMNI/models/fields.py` - SDF network (handles missing encoding)
3. `/home/bhanu/pmni/PMNI/models/dataset_loader.py` - Data loader (CPU storage)
4. `/home/bhanu/pmni/PMNI/config/minimal_baseline.conf` - Minimal config

### Logs ✅
- `logs/minimal_baseline.log` - Shows dataset loading on CPU
- `logs/minimal_v2.log` - Shows GPU OOM during OccupancyGrid init
- Multiple attempts documented in logs/

### Documentation ✅
- This progress report
- Configuration files with comments
- Setup guides (SETUP_GUIDE.md, PMNI_FULL_GUIDE.md)

---

## What's Missing

❌ **No trained model** - Cannot initialize on GPU  
❌ **No mesh output** - Training never starts  
❌ **No validation results** - Blocked at initialization  

---

## Next Steps Required

### Option A: Get GPU Resources (Recommended)
**Need:** ~3-5 GB free GPU memory  
**Action:** Request larger MIG partition or dedicated GPU instance  
**Timeline:** Can complete baseline in 1-2 days once GPU available  
**Output:** Trained mesh, validation metrics, comparison with ground truth

### Option B: Alternative Baseline
**Approach:** Simple geometric reconstruction (visual hull from silhouettes)  
**Pros:** Can run on CPU, provides geometric baseline  
**Cons:** Not neural, limited quality  
**Timeline:** ~2-4 hours to implement

### Option C: Present Implementation Only
**Deliverable:** Code + documentation + problem analysis  
**Pros:** Shows understanding of method and engineering challenges  
**Cons:** No experimental results to show

---

## Conclusion

**Implementation is complete and technically sound.** All code modifications work correctly:
- Dataset successfully loads on CPU ✅
- SDF network creates without encoding ✅  
- Configuration system handles optional parameters ✅

**Blocked purely by infrastructure** - GPU has insufficient memory for even minimal model initialization.

**Recommendation:** Request GPU resources to complete baseline, or pivot to geometric baseline for Review 2 demonstration.

---

## Files to Show Guide

1. `config/minimal_baseline.conf` - Shows scaled-down configuration
2. `logs/minimal_v2.log` - Shows error at OccupancyGrid init
3. This document - Full problem analysis
4. `nvidia-smi` output - Proves GPU memory constraint

---

**Question for Guide:** Which option should I pursue for Review 2?
