# PMNI – End‑to‑End Process and File-by-File Guide

This document explains, step by step, how the project runs (from setup to outputs) and describes the purpose of every important file and folder in this repo. It’s designed as a single, comprehensive reference.

---

## 1) What PMNI does (high level)

PMNI reconstructs a 3D surface (mesh) from multi‑view normal maps using an SDF (signed distance function) network. It:

1. Loads multi‑view normals, masks, camera intrinsics, and initial poses
2. Optimizes an SDF network with volumetric rendering and several losses
3. Optionally optimizes camera pose deltas and per‑view depth scales
4. Periodically validates by rendering normals and extracting a mesh
5. Saves results under an experiment folder per object

Core ideas:
- SDF MLP with multi‑resolution hash encoding (tiny-cuda-nn)
- Ray marching and occupancy grid (nerfacc)
- Photometric/geometry losses: normal, depth, mask, eikonal, etc.
- Pose and scale learning modules

---

## 2) End‑to‑end process (step by step)

### A. Environment and data

- Create the environment and install dependencies:
  - Script: `setup_clean.sh` (CUDA 12.1 flow: PyTorch 2.1.0 + tiny-cuda-nn + pytorch3d + nerfacc + libs)
- Arrange the dataset as expected (see dataset section below). Example configs are in `config/`.

### B. Launch training

- Entry point: `exp_runner.py`
- Typical command (Linux/bash):
  - Single object (bear):
    ```bash
    CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear
    ```
  - All DiLiGenT objects (use `config/diligent.conf`):
    ```bash
    CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent.conf
    ```
- Optional convenience wrapper: `run_training.sh` (logs to `logs/train_<timestamp>.log`).

### C. Initialization inside `exp_runner.py`

1. Parse `.conf` with pyhocon, create an experiment directory: `${general.base_exp_dir}` with timestamp.
2. Instantiate dataset: `models.dataset_loader.Dataset` using `dataset{...}` section.
3. Build networks:
   - `models.fields.SDFNetwork`: SDF MLP with hash encoding
   - `models.fields.SingleVarianceNetwork`: global variance parameter
   - `models.renderer.NeuSRenderer`: volumetric renderer using nerfacc
4. Optional learning heads (gated by `model.LearnPose`, `model.LearnScale`):
   - `models.pose_net.LearnPose`: learn per‑camera delta pose (SE3) from init
   - `models.scale_net.LearnScale`: learn per‑camera multiplicative depth scale
5. Optimizers: Adam for SDF/variance (+ pose/scale if enabled).
6. Schedules: cosine decay for learning rates; staged loss weights (s1/s2/s3) by iteration.
7. Bookkeeping: tensorboard writer, checkpoints, config backup into `exp/.../recording/`.

### D. Training loop (per iteration)

1. Adjust ray marching step size and update occupancy grid.
2. Increase SDF encoding bandwidth every N iters (progressive hash levels).
3. For each view i in [0..N-1]:
   - Sample pixel patches with current pose i: `Dataset.gen_random_patches_fixed_idx_pose(...)`
   - Compute near/far intersections with unit sphere bound
   - Renderer marches rays, accumulates weights, computes:
     - `comp_normal` (rendered normals)
     - `comp_depth` (ray mid‑point depths)
     - `weight_sum` (opacity mask)
     - `gradients` (SDF gradients) for eikonal
     - Optional visibility/consistency features when `con` stage is active
   - Prepare depth inputs (optionally scaled by `LearnScale`), and reference view pairs if point‑cloud consistency is used
   - Evaluate losses via `models.losses.Loss` with staged weights:
     - Normal loss (L1/L2)
     - Depth loss (scale‑invariant patch loss)
     - Mask loss (BCE on rendered occupancy vs. segmentation mask)
     - Eikonal loss (||∇SDF||→1)
     - SDF loss (optional) on sampled points
     - Cross‑view consistency loss (optional) using visibility
4. Accumulate loss over all views, backpropagate, and step optimizers (SDF/var/pose/scale).
5. Logging: scalars to TensorBoard every step, periodic console summaries.

### E. Validation and outputs

- Pose snapshots: `exp/.../poses/<iter>.npy`
- Normal/depth rendering: `validate_normal_pixel_based` every `val.val_normal_freq` → `exp/.../normals/*.png`, `exp/.../depth/*.npy`
- Mesh extraction: `validate_mesh` every `val.val_mesh_freq` via marching cubes → `exp/.../meshes_validation/iter_XXXXXXXX.ply`
- Geometry eval points and OBJ export: `eval_geo` → `exp/.../points_val/`
- CSV metrics (if GT available): `eval_metrics.csv`
- Checkpoints: `exp/.../checkpoints/ckpt_XXXXXX.pth`

---

## 3) Dataset expectations

From configs in `config/*.conf`:
- `dataset.data_dir`: root for each object (e.g., `data/diligent_mv_normals/bear/`)
- Each object folder typically contains:
  - `normal_camera_space_GT/*.exr` or `normal_camera_space_sdmunips/*.exr`
  - `mask/*.png`
  - `integration/depth/*.npy` and `integration/perspective/*.npy`
  - `K.txt` and `cameras_sphere.npz` (intrinsics/poses)
  - Optionally `mesh_Gt.ply`

Loading details (`models/dataset_loader.py`):
- Reads normals, flips Y/Z for camera→world alignment; loads masks and intrinsics; builds `pose_all` from `cameras_sphere.npz`.
- Optionally injects pose noise.
- Provides helpers to sample rays/patches and compute near/far against a unit sphere.

---

## 4) Configuration files (how they drive training)

Configs use HOCON format (parsed by pyhocon). Key blocks:

- `general{}`
  - `dataset_class`, `renderer_class`
  - `base_exp_dir`: where outputs are saved (with an auto timestamp)
  - `pose_init_radius`, `pose_init_angle`, `clock_wise`: for spherical init

- `dataset{}`
  - `data_dir`, `normal_dir`, `cameras_name`
  - `depth_dir`, `perspective_dir`
  - `exclude_views`, `upsample_factor`, `depth_init_scale`

- `train{}`
  - `learning_rate`, `pose_lr`, `scale_lr`, `end_iter`, `warm_up_end`
  - `batch_size`, `patch_size`, `pc_ratio`
  - Loss types and staged weights: `*_weight_s1/s2/s3`

- `val{}`
  - Frequencies for saving, rendering normals, mesh extraction, metrics

- `model{}`
  - `sdf_network{}`: MLP shape and init
  - `variance_network{}`: single learnable variance
  - `ray_marching{}`: step sizes, occupancy thresholds
  - `encoding{}`: tiny‑cuda‑nn hash grid parameters
  - `LearnScale{}`, `LearnPose{}`: toggles and options

Examples:
- `config/diligent_bear.conf`: single object (bear)
- `config/diligent.conf`: all DiLiGenT objects
- `config/own_objects.conf`: custom objects
- `config/ball.conf`, `config/cylinder.conf`: synthetic shapes

---

## 5) File‑by‑file walkthrough

Top‑level:
- `exp_runner.py`: Main training script. Orchestrates dataset, networks, losses, training loop, validation, and I/O.
- `run_training.sh`: Convenience launcher that activates env and logs to `logs/`.
- `setup_clean.sh`: Automated environment setup for CUDA 12.1 systems.
- `README.md`: Project intro and quick start.
- `SETUP_GUIDE.md`: Detailed environment/data setup and troubleshooting.
- `requirements.txt`: Minimal pip dependencies (pip/venv route).

Configs and data:
- `config/*.conf`: HOCON configurations that control everything listed above.
- `data/`: Datasets. Subfolders for DiLiGenT, own objects, synthetic cases.
- `exp/`: Training outputs per experiment.
- `fig/`: Figures (optional, for visualization).
- `logs/`: TensorBoard event files, shell logs.

Core model code (`models/`):
- `models/fields.py`
  - `SDFNetwork`: SDF MLP with optional hash encoding; progressive bandwidth; exposes `sdf`, `gradient`.
  - `SingleVarianceNetwork`: single learnable variance used in the NeuS‑style alpha formulation.
- `models/renderer.py`
  - `NeuSRenderer`: volumetric renderer using nerfacc. Handles ray marching (patch & pixel based), occupancy grid, normal/depth composition, mesh extraction.
  - Also computes visibility and cross‑view data when consistency losses are enabled.
- `models/dataset_loader.py`
  - Loads normals/masks/depth/perspective/intrinsics/poses, optional upsampling and pose noise.
  - Provides ray/patch sampling utilities used by training and validation.
- `models/losses.py`
  - Implements normal/depth/mask/eikonal/SDF and optional cross‑view consistency losses; also a basic point‑cloud consistency helper.
- `models/pose_net.py`
  - `LearnPose`: per‑camera delta pose in axis‑angle + translation; composes with init pose to produce `c2w` matrices.
- `models/scale_net.py`
  - `LearnScale`: per‑camera multiplicative depth scale (with optional fixed scale for the last camera).
- `models/init_pose.py`
  - Sampling points on a view sphere and generating `c2w` with `pytorch3d` look‑at; used for initializing poses.
- `models/common.py`
  - Pixel grid helpers, camera projection/back‑projection, SE(3) utilities, simple viz helpers.
- `models/rend_util.py`
  - Camera projection math, quaternion helpers, sphere intersection utilities for visibility tracing.
- `models/visibility_tracer.py`
  - Sphere tracing from points to cameras using the SDF to compute visibility masks.

Utilities:
- `utilities/utils.py`: Image cropping, angular error maps, simple video and RGBA helpers.

Third‑party:
- `third_parties/nerfacc-0.3.5/`: Local copy for installation (`setup_clean.sh` installs it in editable mode).

---

## 6) How training evolves (stages and schedules)

- Iteration‑based stages switch loss weights:
  - s1 (< 10k): focus on depth/mask/eikonal
  - s2 (10k–20k): add normal loss with higher weights; still depth/mask/eikonal
  - s3 (≥ 20k): emphasize normals; depth may be reduced to 0; optional consistency (`con`) can be enabled in code
- Learning rate schedule: warm‑up to cosine decay for SDF; separate schedules for pose/scale when enabled.
- Hash encoding bandwidth gradually increases every `train.increase_bindwidth_every` steps.

---

## 7) Outputs and where to find them

Within `exp/<group>/<object>/<exp_timestamp>/`:
- `checkpoints/ckpt_XXXXXX.pth`: SDF + variance + optimizer states
- `recording/`: backup of Python sources and the exact config used
- `normals/*.png`: rendered normals (raw and normalized) per validation step
- `depth/*.npy`: rendered depth maps
- `meshes_validation/iter_XXXXXXXX.ply`: marching‑cubes meshes
- `points_val/`: eval points and exported OBJ
- `poses/<iter>.npy`: all camera `c2w` at that iteration
- `eval_metrics.csv`: MAE (and optional geometry metrics)

TensorBoard event files are placed under top‑level `./logs/` (see `exp_runner.py` writer path).

---

## 8) Practical tips

- GPU memory: reduce `train.batch_size` or increase `model.ray_marching.end_step_size` if OOM.
- Dataset paths: ensure `dataset.data_dir` matches your object folder names (`CASE_NAME` is replaced by `--obj_name`).
- Normals domain: if using camera‑space normals, they are rotated to world space in the dataset; ensure the chosen `normal_dir` matches your data.
- Pose init: `general.pose_init_angle/radius` guide initial camera positions on a sphere if `LearnPose.init_pose` is true.

---

## 9) Try it quickly

- Single object:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear
  ```
- Custom object (e.g., `cat`):
  ```bash
  CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/own_objects.conf --obj_name cat
  ```
- Resume style behavior: pass `--is_continue` to load the latest checkpoint (uses the current run’s exp dir if present).

---

## 10) Glossary (key tensors)

- `pose_all` (N, 4, 4): initial camera to world per view
- `LearnPose(idx)` → `c2w` (4×4): current learned camera to world for view idx
- `comp_normal` (B, H_p, W_p, 3): rendered normal per patch
- `comp_depth` (B, H_p, W_p, 1): mid‑point ray depth per patch
- `weight_sum` (B, H_p, W_p, 1): accumulated opacity per pixel
- `gradients` (… , 3): SDF gradients at sample points

---

## 11) License and acknowledgements

As noted in `README.md`, this implementation builds upon:
- SuperNormal, NoPe‑NeRF, BARF

And uses these libraries heavily: PyTorch, tiny‑cuda‑nn, nerfacc, pytorch3d, open3d, trimesh, pyvista, pyexr, etc.

---

If you spot any mismatch between the guide and the code, open an issue or update this file to keep it accurate.
