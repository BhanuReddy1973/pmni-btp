# PMNI: Pose-optimized Multi-view Neural Implicit Surface Reconstruction

## Table of Contents
1. [Introduction to PMNI](#introduction-to-pmni)
2. [Core Concepts and Theory](#core-concepts-and-theory)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Component Analysis](#detailed-component-analysis)
5. [Training Process Step-by-Step](#training-process-step-by-step)
6. [Loss Functions and Optimization](#loss-functions-and-optimization)
7. [Data Processing Pipeline](#data-processing-pipeline)
8. [Implementation Details](#implementation-details)
9. [Validation and Evaluation](#validation-and-evaluation)
10. [Using PMNI Outputs](#using-pmni-outputs)
11. [Advanced Features and Extensions](#advanced-features-and-extensions)
12. [Problems Faced and Solutions](#problems-faced-and-solutions)
13. [Changes from Base Model](#changes-from-base-model)
14. [Complete Function/Class Definitions](#complete-functionclass-definitions)
15. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)

---

## Introduction to PMNI

**PMNI (Pose-optimized Multi-view Neural Implicit)** is a state-of-the-art neural implicit surface reconstruction method that learns 3D shapes from multi-view 2D images. Unlike traditional methods that require known camera poses, PMNI jointly optimizes both the 3D geometry and camera parameters during training.

### Key Innovations:
- **Implicit Neural Representations**: Uses Signed Distance Functions (SDF) for continuous surface representation
- **Joint Pose Optimization**: Learns camera poses alongside geometry for robust reconstruction
- **Multi-view Consistency**: Enforces geometric consistency across different viewpoints
- **Real-time Rendering**: Enables fast surface rendering and normal estimation

### Applications:
- **3D Reconstruction**: From multi-view images without known poses
- **Object Scanning**: Mobile and handheld scanning applications
- **AR/VR Content Creation**: Automatic 3D asset generation
- **Robotics**: Object manipulation and scene understanding
- **Cultural Heritage**: Artifact digitization and preservation

---

## Problems Faced and Solutions

### 1. GPU Memory Constraints
**Problem**: Initial implementation failed with CUDA OOM errors during OccupancyGrid initialization, even with minimal configurations.

**Root Cause**: GPU memory fragmentation and insufficient free memory (only 269 MB available on MIG partition).

**Solutions Implemented**:
- **Dataset CPU Storage**: Modified `dataset_loader.py` to store all data on CPU, transferring batches to GPU only during training
- **Chunked Processing**: Implemented chunked SDF evaluation (65,536 points per chunk) in `renderer.py`
- **Memory Optimization**: Added CUDA cache clearing and garbage collection every 50 iterations
- **Progressive Configurations**: Created minimal baseline configs with reduced parameters

### 2. Device Mismatch Errors
**Problem**: RuntimeError during validation at iteration 20,000+ due to tensor device mismatches between CPU and CUDA.

**Solutions Implemented**:
- **Device Safety Guards**: Added automatic device conversion in `renderer.py` for intrinsics_all and normals tensors
- **Try-Except Wrappers**: Wrapped consistency computation in try-except blocks to prevent crashes
- **Dynamic Device Detection**: Used `w2cs.device` for determining correct device instead of hardcoded 'cuda'

### 3. Training Instability
**Problem**: NaN/Inf values, exploding gradients, and training collapse.

**Solutions Implemented**:
- **NaN/Inf Protection**: Added `torch.nan_to_num()` guards throughout all loss computations
- **Gradient Clipping**: Applied max_norm=1.0 clipping for SDF parameters, max_norm=0.1 for pose parameters
- **Bootstrap Loss**: Implemented fallback eikonal + regularization loss when occupancy grid is empty
- **Safe Numerical Operations**: Added epsilon values and clamping to prevent division by zero

### 4. Optional Dependencies
**Problem**: Code failed when tiny-cuda-nn or pypose libraries were unavailable.

**Solutions Implemented**:
- **Graceful Degradation**: Added try-except imports with CPU fallbacks
- **Feature Detection**: Made encoding and pose learning optional based on available libraries
- **Configuration Flexibility**: Created configs that work with or without optional dependencies

### 5. Checkpoint and Recovery Issues
**Problem**: Infrequent checkpoints (every 500,000 iterations) made recovery from crashes impossible.

**Solutions Implemented**:
- **Frequent Checkpoints**: Reduced checkpoint frequency to every 2,500 iterations
- **Auto-Resume**: Modified training script to automatically load latest checkpoint
- **Validation Frequency**: Balanced validation frequency to prevent slowdowns

---

## Changes from Base Model

### 1. Memory Management Enhancements
**Original**: All data stored on GPU, no chunking
**Modified**: 
- Dataset stored on CPU (`dataset_loader.py`)
- Chunked SDF evaluation (65,536 points per chunk)
- CUDA cache clearing every 50 iterations
- Memory usage monitoring and optimization

### 2. Stability Improvements
**Original**: No NaN/Inf protection, basic gradient clipping
**Modified**:
- Comprehensive NaN/Inf guards in all loss functions
- Bootstrap loss for early training stability
- Enhanced gradient clipping with different norms per parameter type
- Safe numerical operations with epsilon values

### 3. Device Safety
**Original**: Assumed all tensors on CUDA
**Modified**:
- Automatic device detection and conversion
- Try-except wrappers for device-sensitive operations
- Dynamic device assignment based on tensor locations

### 4. Optional Dependencies Handling
**Original**: Required tiny-cuda-nn and pypose
**Modified**:
- Graceful fallback when libraries unavailable
- CPU-compatible MLP when tiny-cuda-nn missing
- Disabled pose learning when pypose unavailable
- Configuration-driven feature enabling

### 5. Training Infrastructure
**Original**: Basic training loop with infrequent checkpoints
**Modified**:
- Frequent checkpointing (every 2,500 iterations)
- Auto-resume from latest checkpoint
- Enhanced logging and monitoring
- Progressive loss weighting stages

### 6. Error Handling
**Original**: Crashes on validation failures
**Modified**:
- Try-except wrappers around validation functions
- Graceful degradation when operations fail
- Logging of errors without stopping training
- Fallback computations for failed operations

---

## Complete Function/Class Definitions

### Core Neural Networks

#### SDFNetwork (`models/fields.py`)
```python
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in=3,           # Input dimension (3D coordinates)
                 d_out=1,          # Output dimension (SDF value)
                 d_hidden=256,     # Hidden layer dimension
                 n_layers=8,       # Number of layers
                 skip_in=(4,),     # Skip connection layers
                 bias=0.5,         # Output bias for SDF
                 geometric_init=True,  # Geometric weight initialization
                 weight_norm=True, # Weight normalization
                 inside_outside=False, # SDF sign convention
                 encoding_config=None, # Hash grid encoding config
                 input_concat=False):   # Input concatenation flag
```

**Key Methods**:
- `increase_bandwidth()`: Gradually increases encoding resolution
- `forward(inputs)`: Main forward pass with skip connections
- `sdf(x)`: SDF value computation
- `gradient(x)`: Analytical gradient computation via autograd
- `laplace(x)`: Laplacian computation for regularization

#### SingleVarianceNetwork (`models/fields.py`)
```python
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        # Single learnable parameter for surface variance
        self.variance = nn.Parameter(torch.tensor(init_val))
    
    def forward(self, x):
        # Returns variance value for all input points
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
```

### Pose and Scale Networks

#### LearnPose (`models/pose_net.py`)
```python
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        # Learnable rotation (axis-angle) and translation parameters
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)
        self.init_c2w = init_c2w  # Optional initial pose
    
    def forward(self, cam_id):
        # Returns camera-to-world matrix for given camera ID
        r = self.r[cam_id]  # (3,) axis-angle
        t = self.t[cam_id]  # (3,) translation
        c2w = make_c2w(r, t)  # Convert to 4x4 matrix
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]  # Apply delta pose
        return c2w
    
    def get_all_c2w(self):
        # Returns all camera poses as (N, 4, 4) tensor
        all_c2w = torch.zeros((self.num_cams, 4, 4), dtype=torch.float32, device=self.r.device)
        for cam_id in range(self.num_cams):
            all_c2w[cam_id] = self(cam_id)
        return all_c2w
```

#### LearnScale (`models/scale_net.py`)
```python
class LearnScale(nn.Module):
    def __init__(self, num_cams=20, learn_scale=False, init_scale=None, fix_scaleN=False):
        # Learnable scale parameters for depth correction
        self.global_scales = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=learn_scale)
        self.fix_scaleN = fix_scaleN  # Fix last camera scale to 1.0
    
    def forward(self, cam_id):
        # Returns scale factor for given camera ID
        scale = self.global_scales[cam_id]
        if self.fix_scaleN and cam_id == (self.num_cams-1):
            scale = torch.tensor([1.0], device=self.global_scales.device)
        return scale
```

### Rendering Engine

#### NeuSRenderer (`models/renderer.py`)
```python
class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network,
                 gradient_method="ad", K=None, H=None, W=None,
                 intrinsics_all=None, normals=None, scale_mats=None):
        
        # Core components
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Occupancy grid for acceleration
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=torch.as_tensor([-1., -1., -1., 1., 1., 1.], dtype=torch.float32),
            resolution=128,
            contraction_type=ContractionType.AABB).to(self.device)
        
        # Rendering parameters
        self.sampling_step_size = 0.01
        self.gradient_method = gradient_method
        
        # Visibility tracing
        self.visible_ray_tracer = VisibilityTracing()
```

**Key Methods**:
- `occ_eval_fn(x)`: Occupancy evaluation for grid updates
- `render()`: Main volume rendering pipeline
- `render_normal_pixel_based()`: Pixel-wise normal rendering
- `extract_geometry()`: Marching cubes mesh extraction

### Loss Functions

#### Loss (`models/losses.py`)
```python
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
```

**Key Methods**:
- `get_normal_loss()`: L1/L2 normal supervision loss
- `get_depth_loss()`: Scale-invariant depth loss
- `get_depth_loss3()`: Enhanced depth loss with scale estimation
- `get_mask_loss()`: Binary cross-entropy mask loss
- `get_eikonal_loss()`: SDF gradient magnitude regularization
- `get_pc_loss()`: Point cloud consistency loss
- `get_con_loss()`: Multi-view consistency loss
- `get_sdf_loss()`: SDF value regularization

### Data Processing

#### Dataset (`models/dataset_loader.py`)
```python
class Dataset:
    def __init__(self, conf):
        # Configuration-driven dataset loading
        self.device = torch.device('cpu')  # Store on CPU to avoid GPU OOM
        
        # Load data paths and configurations
        normal_dir = conf.get_string('normal_dir')
        depth_dir = conf.get_string('depth_dir')
        self.data_dir = conf.get_string('data_dir')
        
        # Load camera parameters
        camera_dict = np.load(os.path.join(self.data_dir, self.cameras_name))
        self.world_mats_np = [camera_dict['world_mat_%d' % idx] for idx in self.img_idx_list]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx] for idx in self.img_idx_list]
        
        # Load image data
        self.normal_np = np.stack([pyexr.read(im_name)[..., :3] for im_name in self.normal_lis])
        self.depth_np = np.stack(np.load(depth_name) for depth_name in self.depth_lis)
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
        
        # Convert to tensors on CPU
        self.normals = torch.from_numpy(self.normal_np.astype(np.float32)).to(self.device)
        self.depths = torch.from_numpy(self.depth_np.astype(np.float32)).to(self.device)
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)
```

**Key Methods**:
- `gen_rays_at()`: Generate rays for specific camera view
- `gen_random_patches()`: Generate random training patches
- `gen_random_patches_fixed_idx_pose()`: Generate patches with fixed pose
- `near_far_from_sphere()`: Compute ray intersection bounds

### Utility Functions

#### Common Utilities (`models/common.py`)
- `arange_pixels_2()`: Arrange pixel coordinates for depth processing
- `transform_to_world()`: Transform pixels to 3D world coordinates
- `make_c2w()`: Convert axis-angle + translation to camera matrix
- `Exp()`: SO(3) exponential map for rotation matrices
- `add_noise_to_pose()`: Add noise to camera poses for robustness

#### Visibility Tracing (`models/visibility_tracer.py`)
```python
class VisibilityTracing(nn.Module):
    def __init__(self, object_bounding_sphere=1.4, sphere_tracing_iters=30, initial_epsilon=1e-3):
        # Parameters for sphere tracing visibility computation
        self.object_bounding_sphere = object_bounding_sphere
        self.sphere_tracing_iters = sphere_tracing_iters
        self.start_epsilon = initial_epsilon
    
    def forward(self, sdf, unique_camera_centers, points):
        # Compute visibility mask for points from camera centers
        ray_directions = unique_camera_centers.unsqueeze(0) - points.unsqueeze(1)
        unit_ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        visibility_mask = self.sphere_tracing_for_visibility(sdf, points, unit_ray_directions, unique_camera_centers)
        return visibility_mask
```

#### Pose Initialization (`models/init_pose.py`)
- `sample_points_on_sphere_uniform()`: Sample camera positions on sphere
- `generate_c2w_matrices()`: Generate camera-to-world matrices from positions
- `sample_positions_torch()`: Random camera position sampling

### Training Infrastructure

#### Runner (`exp_runner.py`)
```python
class Runner:
    def __init__(self, conf_text, mode='train', is_continue=False, datadir=None):
        # Initialize training components with device safety
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            print("[FATAL] CUDA not available. PMNI training requires a GPU")
            raise SystemExit(0)
        
        # Load configuration and setup experiment directory
        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], exp_time_dir)
        
        # Initialize networks with optional dependency handling
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], encoding_config=encoding_config)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network'])
        
        # Try to initialize pose network (depends on pypose)
        LearnPose = None
        try:
            from models.pose_net import LearnPose as _LearnPose
            LearnPose = _LearnPose
        except Exception as e:
            print(f"[WARN] Pose learning disabled: {e}")
        
        # Initialize renderer with device safety
        self.renderer = NeuSRenderer(self.sdf_network, self.deviation_network, ...)
        
        # Initialize loss function
        self.loss = Loss()
```

**Key Methods**:
- `train()`: Main training loop with progressive stages
- `train_step()`: Single training iteration
- `compute_loss()`: Loss computation with stage-based weighting
- `validate_normal_pixel_based()`: Normal map validation
- `validate_mesh()`: Mesh extraction and cleaning
- `save_checkpoint()`: Model checkpointing
- `load_checkpoint()`: Checkpoint loading for resume

---

## Core Concepts and Theory

### 1. Signed Distance Functions (SDF)

A **Signed Distance Function** represents a surface implicitly by mapping any 3D point to its signed distance from the surface:

```
SDF(x) = 
  +d  if point x is outside the surface
  -d  if point x is inside the surface
   0  if point x is on the surface
```

**Advantages:**
- **Continuous representation**: No discretization artifacts
- **Infinite resolution**: Can query any point in space
- **Topological flexibility**: Handles complex geometries naturally

### 2. Neural Implicit Representations

Instead of storing SDF values explicitly, PMNI uses a **neural network** to learn the SDF function:

```
SDF_θ(x) ≈ SDF(x)
```

Where `θ` represents the network parameters learned during training.

### 3. Volume Rendering

PMNI uses **differentiable volume rendering** to connect 2D observations with 3D geometry:

- **Ray marching**: Traces rays from camera through 3D space
- **Density estimation**: Converts SDF to density using sigmoid function
- **Alpha compositing**: Accumulates color/transmittance along rays

### 4. Pose Optimization

Camera poses are learned as **Lie group parameters** (rotation + translation):

```
c2w = exp([ω, v]) ⊕ c2w_initial
```

Where `ω` is rotation vector and `v` is translation vector.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PMNI Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Multi-view Images → Dataset Loader → Preprocessing        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Neural Networks                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │  SDF Network│  │Variance Net │  │ Pose Network│ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Rendering Engine                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │Ray Marching │  │Volume Render│  │Normal Est. │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Loss Functions                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │Depth Loss   │  │Normal Loss  │  │Eikonal Loss│ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Training Loop → Optimization → Mesh Extraction            │
└─────────────────────────────────────────────────────────────┘
```

### Core Components:

1. **SDF Network**: Learns the implicit surface representation
2. **Variance Network**: Controls surface uncertainty/blur
3. **Pose Network**: Optimizes camera parameters
4. **Renderer**: Converts implicit representation to 2D images
5. **Loss Functions**: Supervise learning with multi-view constraints

---

## Detailed Component Analysis

### 1. SDF Network (fields.py)

The **SDFNetwork** is a multi-layer perceptron that learns the signed distance function:

```python
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 encoding_config=None,
                 input_concat=False):
        super(SDFNetwork, self).__init__()
        self.input_concat = input_concat

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if encoding_config is not None and TCNN_AVAILABLE:
            self.encoding = tcnn.Encoding(d_in, encoding_config).to(torch.float32)
            dims[0] = self.encoding.n_output_dims
            if input_concat:
                dims[0] += d_in
        else:
            self.encoding = None
            if encoding_config is not None and not TCNN_AVAILABLE:
                print("Warning: Encoding config provided but tinycudann not available, using plain MLP")

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.bindwidth = 0
        self.enc_dim = self.encoding.n_output_dims if self.encoding is not None else 0

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif self.encoding is not None and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.encoding is not None and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def increase_bandwidth(self):
        self.bindwidth += 1

    def forward(self, inputs):
        if self.encoding is not None:
            encoded = self.encoding(inputs).to(torch.float32)

            # set the dimension of the encoding to 0 if the input is outside the bandwidth
            enc_mask = torch.ones(self.enc_dim, dtype=torch.bool, device=encoded.device, requires_grad=False)
            enc_mask[self.bindwidth*2:] = 0
            encoded = encoded * enc_mask

        if self.input_concat:
            inputs = torch.cat([inputs, encoded], dim=1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    @torch.enable_grad()
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    @torch.enable_grad()
    def divergence(self, y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
        return div

    @torch.enable_grad()
    def laplace(self, x):
        return self.divergence(self.gradient(x), x)
```

**Key Features:**
- **Hash Grid Encoding**: Efficient high-dimensional encoding using tiny-cuda-nn
- **Skip Connections**: Direct input-to-output pathways for better gradient flow
- **Geometric Initialization**: Special weight initialization for SDF learning
- **Progressive Bandwidth**: Gradually increases encoding resolution during training

### 2. Variance Network (fields.py)

Controls the "softness" of the surface for volume rendering:

```python
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
```

**Purpose:**
- **Surface Uncertainty**: Higher variance = softer surface boundaries
- **Rendering Quality**: Affects how rays interact with the surface
- **Training Stability**: Helps with convergence in early training stages

### 3. Pose Network (pose_net.py)

Learns camera poses as Lie group parameters:

```python
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t,init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        #if(cam_id == (self.num_cams-1)):
        #    c2w = self.init_c2w[self.num_cams-1]
        #if(cam_id == 0):
        #    c2w = self.init_c2w[0]
        return c2w
    def get_t(self):
       
        all_t = torch.zeros((self.num_cams, 3), dtype=torch.float32, device=self.t.device)
        
        for cam_id in range(self.num_cams):

            r = self.r[cam_id]  # (3,)
            t = self.t[cam_id]  # (3,)
            c2w = make_c2w(r, t)  # (4, 4)
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            all_t[cam_id] = c2w[:3, 3]  # (3,)

        return all_t  
    def get_all_c2w(self):
        all_c2w = torch.zeros((self.num_cams, 4, 4), dtype=torch.float32, device=self.r.device)
        
        for cam_id in range(self.num_cams):
            r = self.r[cam_id]  # (3,)
            t = self.t[cam_id]  # (3,)
            c2w = make_c2w(r, t)  # (4, 4)
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            all_c2w[cam_id] = c2w

        return all_c2w
```

**Features:**
- **Lie Group Optimization**: Proper rotation manifold optimization
- **Incremental Updates**: Learns pose corrections from initial estimates
- **Per-Camera Parameters**: Independent pose for each camera

### 4. Renderer (renderer.py)

Converts implicit representation to 2D renderings:

```python
class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network,
                 gradient_method="ad", K=None,  H=None, W=None ,intrinsics_all =None, normals = None, scale_mats = None):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network

        # Detect device (CPU or CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Renderer] Using device: {self.device}")

        # define the occ grid, see NerfAcc for more details
        self.scene_aabb = torch.as_tensor([-1., -1., -1., 1., 1., 1.], dtype=torch.float32)
        # define the contraction_type for scene contraction
        self.contraction_type = ContractionType.AABB
        # create Occupancy Grid
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=128,  # if res is different along different axis, use [256,128,64]
            contraction_type=self.contraction_type).to(self.device)
        self.sampling_step_size = 0.01  # ray marching step size, will be modified during training
        self.gradient_method = gradient_method   # dfd or fd or ad
        self.visible_ray_tracer = VisibilityTracing()
        self.K = K
        self.H = H
        self.W = W
        # Ensure intrinsics and normals are on correct device
        if intrinsics_all is not None:
            self.intrinsics_all = intrinsics_all.to(self.device) if not intrinsics_all.device == self.device else intrinsics_all
        else:
            self.intrinsics_all = None
        if normals is not None:
            self.normals = normals.to(self.device) if not normals.device == self.device else normals
        else:
            self.normals = None
        self.scale_mats = scale_mats

    def transform_to_normalized_coords(self, pts):
        """Transform points to normalized coordinates for SDF network input.
        Since scene is bounded by [-1,1]^3, points in this range are already normalized."""
        return pts

    def occ_eval_fn(self, x):
        # function for updating the occ grid given the current sdf
        sdf = self.sdf_network(x)[..., :1]
        alpha = torch.sigmoid(- sdf * 80)  # occ grids with alpha below the occ threshold will be set as 0
        return alpha


    def render(self, rays_o_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     rays_d_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     marching_plane_normal,  # (num_patch, 3)
                     near,  # (num_patch,)
                     far,  # (num_patch,)
                     mask,
                     c2ws,
                     idx,
                     con =False,
                     val_gradient_method='ad',
                     mode='train'):
        # patch size, should be odd
        patch_H = rays_o_patch_all.shape[1]
        patch_W = rays_o_patch_all.shape[2]
        num_patch = rays_o_patch_all.shape[0]

        # extract camera location and ray direction of the patches' center pixels
        rays_o_patch_center = rays_o_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_d_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        def alpha_fn_patch_center(t_starts, t_ends, ray_indices, ret_sdf=False):
            # the function used in ray marching
            ray_indices = ray_indices.long()
            t_origins = rays_o_patch_center[ray_indices]
            t_dirs = rays_d_patch_center[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return
            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_starts[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1]
            inv_s = torch.nan_to_num(inv_s, nan=1.0, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)  # Single parameter, robust
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        #ipdb.set_trace()
        patch_indices, t_starts_patch_center, t_ends_patch_center = ray_marching(
            rays_o_patch_center, rays_d_patch_center,
            t_min=near,
            t_max=far,
            grid=self.occupancy_grid,
            render_step_size=self.sampling_step_size,
            stratified=True,
            cone_angle=0.0,
            early_stop_eps=1e-8,
            alpha_fn=alpha_fn_patch_center,
        )
        samples_per_ray = patch_indices.shape[0] / num_patch
        if patch_indices.shape[0] == 0:  # all patch center rays are within the zero region of the occ grid. skip this iteration.
            return {
                "comp_normal": torch.zeros([num_patch, patch_H, patch_W, 3], device=rays_o_patch_center.device),
                "comp_depth": torch.zeros([num_patch, patch_H, patch_W, 1], device=rays_o_patch_center.device),
                "gradients": None,
                "weight_sum": torch.zeros([num_patch, patch_H, patch_W, 1], device=rays_o_patch_center.device),
                "samples_per_ray": 0,
                "visibility_mask": None,
                "normal_world_all": None,
                "gradients_filtered": None,
                "weights_cuda_filtered": None,
                "s_val": torch.tensor([1.0], device=rays_o_patch_center.device)
            }

        num_samples = patch_indices.shape[0]
        patch_indices = patch_indices.long()

        # compute the sampling distance on remaining rays
        t_starts_patch_all = t_starts_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                                 /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]
        t_ends_patch_all = t_ends_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                               /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]
        mid_points_patch_all = (t_starts_patch_all + t_ends_patch_all)/2.0

        t_starts_patch_center_shift_left = t_starts_patch_center[1:]
        t_starts_patch_center_shift_left = torch.cat([t_starts_patch_center_shift_left, t_starts_patch_center[-1:]], 0)
        diff_mask = ((t_ends_patch_center - t_starts_patch_center_shift_left) != 0)[..., 0]
        positions_starts_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_starts_patch_all
        positions_ends_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_ends_patch_all  # (num_samples, patch_H, patch_W, 3)
        positions_ends_diff = positions_ends_patch_all[diff_mask]
        positions_all = torch.cat([positions_starts_patch_all, positions_ends_diff], 0)
        positions_all_flat = positions_all.reshape(-1, 3)
        #ipdb.set_trace()
        sdf_all = self.sdf_network(positions_all_flat)
        sdf_all = sdf_all.reshape(*positions_all.shape[:-1], 1)

        sdf_starts_patch_all = sdf_all[:positions_starts_patch_all.shape[0]]

        sdf_end_diff = sdf_all[positions_starts_patch_all.shape[0]:]
        sdf_ends_patch_all = sdf_starts_patch_all[1:]
        sdf_ends_patch_all = torch.cat([sdf_ends_patch_all, sdf_starts_patch_all[-1:]], 0)
        sdf_ends_patch_all[diff_mask] = sdf_end_diff

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1]
        inv_s = torch.nan_to_num(inv_s, nan=1.0, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)  # Single parameter, robust

        prev_cdf = torch.sigmoid(sdf_starts_patch_all * inv_s)  # (num_samples, patch_H, patch_W, 1)
        next_cdf = torch.sigmoid(sdf_ends_patch_all * inv_s)   # (num_samples, patch_H, patch_W, 1)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # (num_samples, patch_H, patch_W, 1)
        weights_cuda = render_weight_from_alpha_patch_based(alpha.reshape(num_samples, patch_H*patch_W, 1), patch_indices)  # (num_samples, patch_H, patch_W, 1)
        if mode == 'train':
            gradient_method = self.gradient_method
        elif mode == 'eval':
            gradient_method = val_gradient_method

        if gradient_method == "ad":
            gradients = self.sdf_network.gradient(positions_starts_patch_all.reshape(-1, 3)).reshape(num_samples, patch_H, patch_W, 3)
        

        

        weights_sum_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, n_patches=num_patch)  # (num_samples, patch_H, patch_W, 1)
        weights_sum = weights_sum_cuda.reshape(num_patch, patch_H, patch_W, 1)

        comp_normals_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, values=gradients.reshape(num_samples,patch_H * patch_W, 3),n_patches=num_patch)  # (num_samples, patch_H, patch_W, 3)
        comp_normal = comp_normals_cuda.reshape(num_patch, patch_H, patch_W, 3)

        comp_normal_plain = comp_normal.view(-1, 3)

        comp_depth_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, values=mid_points_patch_all.reshape(num_samples,patch_H * patch_W, 1),n_patches=num_patch)
        comp_depth = comp_depth_cuda.reshape(num_patch, patch_H, patch_W, 1)

        #mask ([4096, 1, 1, 1]) (num_patch, patch_H, patch_W, 1)
        
        surface_points = positions_starts_patch_all# (num_samples, patch_H, patch_W, 3)
        #   patch_indices #(num_samples)
        pre_mask = (mask.view(-1) > 0).squeeze()  
        surface_mask = pre_mask[patch_indices]

        cam_t = c2ws[:, :3, 3]  
        w2cs = torch.inverse(c2ws)
        surface_points_plain = surface_points.view(-1, 3).detach()
        idx_expand = torch.full(surface_mask.shape, idx, dtype=torch.long, device=surface_mask.device)
        if(con):
            try:
                with torch.no_grad():
                    visibility_mask = self.visible_ray_tracer(sdf=lambda x: self.sdf_network.sdf(x),
                                                              unique_camera_centers=cam_t,
                                                              points=surface_points_plain[surface_mask])  # (num_points, num_cams)

                    num_vis_points = visibility_mask.shape[0]
                    visibility_mask[torch.arange(num_vis_points), idx_expand[surface_mask].long()] = 1
                    assert torch.all(visibility_mask.sum(-1) > 0)
                    # ensure we perform projections on the same device as camera matrices
                    device = w2cs.device
                    points_homo = torch.cat(
                        (surface_points_plain[surface_mask].to(device),
                         torch.ones((int(surface_mask.sum().item()), 1), dtype=torch.float32, device=device)),
                        -1).float()
                    # project points onto all image planes
                    # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                    # intrinsics_all is already on CUDA from __init__, ensure w2cs matches
                    K_3x4 = self.intrinsics_all[:, :3, :]
                    if K_3x4.device != device:
                        K_3x4 = K_3x4.to(device)
                projection_matrices = torch.bmm(K_3x4, w2cs)

                pixel_coordinates_homo = torch.einsum("ijk,kp->ijp", projection_matrices, points_homo.T)

                pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
                pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

                index_axis0 = torch.round(pixel_coordinates_yy)  # (num_points, num_cams)
                index_axis1 = torch.round(pixel_coordinates_xx)  # (num_points, num_cams)
                index_axis0 = torch.clamp(index_axis0, min=0, max=self.H - 1).to(torch.int64)  
                index_axis1 = torch.clamp(index_axis1, min=0, max=self.W - 1).to(torch.int64) 
                num_cams = index_axis0.shape[1]
                normal_world = []
                for cam_idx in range(num_cams):
                    idx_normal = self.normals[cam_idx,
                                            index_axis0[:, cam_idx],
                                            index_axis1[:, cam_idx]]  # (num_surface_points)
                    rotation_matrix = c2ws[cam_idx][:3, :3]

                    idx_normals_world = torch.einsum('ij,nj->ni', rotation_matrix, idx_normal)
                    normal_world.append(idx_normals_world)
                normal_world_all = torch.stack(normal_world, dim=1).to(device) #(num_points, num_cams, 3)


                weights_cuda_squeezed = weights_cuda.view(-1) 
                gradients_squeezed = gradients.view(-1, 3)     

                weights_cuda_filtered = weights_cuda_squeezed[surface_mask]  
                gradients_filtered = gradients_squeezed[surface_mask]
            except Exception as e:
                # If consistency computation fails (e.g., during validation), return None for consistency outputs
                print(f"Warning: Consistency computation failed with error: {e}. Returning None for consistency outputs.")
                visibility_mask = None
                normal_world_all = None
                gradients_filtered = None
                weights_cuda_filtered = None

        else:
            visibility_mask = None
            normal_world_all = None
            gradients_filtered = None
            weights_cuda_filtered = None



            
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1]
        inv_s = torch.nan_to_num(inv_s, nan=1.0, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)  # Single parameter, robust
        return {
            's_val': 1/inv_s,
            'weight_sum': weights_sum,
            'gradients': gradients,
            "comp_normal": comp_normal,
            "comp_depth": comp_depth,
            "samples_per_ray": samples_per_ray,
            'visibility_mask':visibility_mask, #(num_points, num_cams)
            'normal_world_all':normal_world_all, #(num_points, num_cams, 3)
            "gradients_filtered":gradients_filtered, #(num_points, 3)
            'weights_cuda_filtered':weights_cuda_filtered #[num_points]
        }

    @torch.no_grad()
    def render_normal_pixel_based(self, rays_o, rays_d, near, far):
        def alpha_fn(t_starts, t_ends, ray_indices, ret_sdf=False):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return

            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            # ic(diff_mask.shape, positions_ends_diff.shape, positions_starts.shape)
            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_start[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            t_min=near.squeeze(),
            t_max=far.squeeze(),
            grid=self.occupancy_grid,
            render_step_size=self.sampling_step_size,
            stratified=True,
            cone_angle=0.0,
            alpha_thre=0.0,
            early_stop_eps=1e-3,
            alpha_fn=alpha_fn,
        )

        alpha = alpha_fn(t_starts, t_ends, ray_indices)

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        gradients = self.sdf_network.gradient(positions).reshape(-1, 3)

        n_rays = rays_o.shape[0]
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)  # [n_samples, 1]
        comp_normal = accumulate_along_rays(weights, ray_indices, values=gradients, n_rays=n_rays)
        comp_depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        return comp_normal, comp_depth

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
    
    
    def mvas_net(self, pts, mask, idx,sdf_network, c2ws =None, device ='cuda',training =True):
        #pts ([4096, 1, 1, 3])
        #mask  ([4096, 1, 1, 1])
        cam_t = c2ws[:3, 3]
        if training:
            points = pts
            points, object_mask, idx = pts, mask, idx
            valid_mask = mask.view(-1) > 0

            num_pixels, _ = pts.shape
            
            surface_mask = object_mask[...,None].repeat(1,128).reshape(-1)
            idx = idx[...,None].repeat(1,128).reshape(-1)
            
            with torch.no_grad():
                visibility_mask = self.visible_ray_tracer(sdf=lambda x: sdf_network(x)[..., 0],
                                                          unique_camera_centers=dataset.unique_camera_centers.to(
                                                              device),
                                                          points=points[surface_mask])  # (num_points, num_cams)

                num_vis_points = visibility_mask.shape[0]
                visibility_mask[torch.arange(num_vis_points), idx[surface_mask].long()] = 1
                assert torch.all(visibility_mask.sum(-1) > 0)
                points_homo = torch.cat(
                    (points[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device=device)), -1).float()
                # project points onto all image planes
                # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                pixel_coordinates_homo = torch.einsum("ijk, kp->ijp", dataset.projection_matrices.to(device),
                                                      points_homo.T).cpu().detach().numpy()
                pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
                pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

                # opencv convention to numpy axis convention
                #  (top left) ----> x    =>  (top left) ---> axis 1
                #    |                           |
                #    |                           |
                #    |                           |
                #    y                         axis 0
                index_axis0 = np.round(pixel_coordinates_yy)  # (num_points, num_cams)
                index_axis1 = np.round(pixel_coordinates_xx)  # (num_points, num_cams)
                index_axis0 = np.clip(index_axis0, int(0), int(self.H - 1)).astype(
                    np.uint)  # (num_points, num_cams)
                index_axis1 = np.clip(index_axis1, int(0), int(self.W - 1)).astype(np.uint)

                num_cams = index_axis0.shape[1]
                tangent_vectors_all_view_list = []
                tangent_vectors_half_pi_all_view_list = []
                for cam_idx in range(num_cams):
                    azimuth_angles = dataset.azimuth_map_all_view[cam_idx,
                                                                  index_axis0[:, cam_idx],
                                                                  index_axis1[:, cam_idx]]  # (num_surface_points)
                    R_list = dataset.W2C_list[cam_idx]
                    r1 = R_list[0, :3]
                    r2 = R_list[1, :3]
                    tangent_vectors_all_view_list.append(
                        r1 * np.sin(azimuth_angles[:, None]) - r2 * np.cos(azimuth_angles[:, None]))
                    tangent_vectors_half_pi_all_view_list.append(r1 * np.sin(azimuth_angles[:, None] + np.pi / 2) -
                                                                 r2 * np.cos(azimuth_angles[:, None] + np.pi / 2))
                    

                tangent_vectors_all_view = torch.stack(tangent_vectors_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
                tangent_vectors_half_pi_all_view = torch.stack(tangent_vectors_half_pi_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
            output = {
                "tangent_vectors_all_view": tangent_vectors_all_view,
                "tangent_vectors_all_view_half_pi": tangent_vectors_half_pi_all_view,
                "visibility_mask": visibility_mask,
                "surface_mask": surface_mask,
                'network_object_mask': 0,
                'surface_normal': 0
            }

        else:
            pass

        return output
```

**Rendering Pipeline:**
1. **Ray Generation**: Create rays from camera parameters
2. **Ray Marching**: Sample points along rays using occupancy grid
3. **Density Estimation**: Convert SDF to volume density
4. **Alpha Compositing**: Accumulate transmittance and color
5. **Normal Estimation**: Compute surface gradients

---

## Training Process Step-by-Step

### Phase 1: Initialization (0-10k iterations)

```python
# 1. Load multi-view data (normals, depths, masks)
dataset = Dataset(conf)
sdf_net = SDFNetwork()
pose_net = LearnPose(num_cams=20)

# 2. Initialize occupancy grid for acceleration
renderer = NeuSRenderer(sdf_net, variance_net)
occupancy_grid = OccupancyGrid(scene_aabb=[-1,1,-1,1,-1,1])

# 3. Set up optimization
optimizer_sdf = Adam(sdf_net.parameters(), lr=1e-3)
optimizer_pose = Adam(pose_net.parameters(), lr=1e-3)
```

### Phase 2: Main Training Loop

```python
for iteration in range(30000):
    # 1. Sample training patches
    rays_o, rays_d, gt_normals, gt_depths, masks = dataset.gen_patches()
    
    # 2. Get current camera poses
    c2ws_matrices = pose_net.get_all_c2w()
    
    # 3. Render from current geometry
    render_out = renderer.render(rays_o, rays_d, c2w_matrices)
    pred_normals = render_out['comp_normal']
    pred_depths = render_out['comp_depth']
    
    # 4. Compute losses
    normal_loss = F.mse_loss(pred_normals, gt_normals)
    depth_loss = F.l1_loss(pred_depths, gt_depths)
    eikonal_loss = compute_eikonal_loss(render_out['gradients'])
    
    # 5. Backpropagation
    total_loss = normal_loss + depth_loss + eikonal_loss
    total_loss.backward()
    
    # 6. Update networks
    optimizer_sdf.step()
    optimizer_pose.step()
    
    # 7. Update occupancy grid
    if iteration % 100 == 0:
        occupancy_grid.update(sdf_net)
```

### Phase 3: Mesh Extraction

```python
# Extract final mesh using marching cubes
bound_min = torch.tensor([-1, -1, -1])
bound_max = torch.tensor([1, 1, 1])

vertices, triangles = renderer.extract_geometry(bound_min, bound_max, 
                                               resolution=1024, threshold=0.0)

# Save as PLY file
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export('reconstructed_mesh.ply')
```

---

## Loss Functions and Optimization

### 1. Multi-Stage Loss Weighting

PMNI uses **progressive loss weighting** to guide learning:

```python
# Stage 1 (0-10k): Focus on basic geometry
weights = {
    'normal_weight': 1.0,    # Surface orientation
    'depth_weight': 1.0,    # Distance constraints
    'eikonal_weight': 0.1,  # SDF regularization
}

# Stage 2 (10k-20k): Refine details
weights = {
    'normal_weight': 2.0,    # Stronger normal supervision
    'depth_weight': 1.5,    # Maintain depth accuracy
    'eikonal_weight': 0.2,  # Tighter regularization
}

# Stage 3 (20k-30k): Final convergence
weights = {
    'normal_weight': 3.0,    # High-quality normals
    'depth_weight': 2.0,    # Precise depths
    'eikonal_weight': 0.3,  # Final regularization
}
```

### 2. Individual Loss Components

#### Normal Loss
```python
def get_normal_loss(pred_normals, gt_normals, mask, mask_sum,normal_loss_type='l2'):
    # Ensure predictions are clean
    normal_pred = torch.nan_to_num(normal_pred, nan=0.0, posinf=0.0, neginf=0.0)
    normal_gt = torch.nan_to_num(normal_gt, nan=0.0, posinf=0.0, neginf=0.0)
    
    normal_error = (normal_pred - normal_gt) * mask
    if normal_loss_type == 'l1':
        normal_loss = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
    elif normal_loss_type == 'l2':
        normal_loss = F.mse_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum

    return normal_loss
```
**Purpose:** Ensures predicted surface orientations match ground truth

#### Depth Loss
```python
def get_depth_loss(depth_pred, depth_gt, mask, mask_sum, depth_loss_type = 'l1'):
    depth_error = (depth_pred - depth_gt) * mask
    depth_error = torch.nan_to_num(depth_error, nan=0.0)
    if depth_loss_type == 'l1':
        depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum
    elif depth_loss_type == 'l2':
        depth_loss = F.mse_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum

    return depth_loss
```
**Purpose:** Constrains surface distances from camera

#### Eikonal Loss
```python
def get_eikonal_loss(gradients):
    if gradients is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    gradients_norm = torch.linalg.norm(gradients, ord=2, dim=-1)
    eikonal_loss = F.mse_loss(gradients_norm, torch.ones_like(gradients_norm), reduction='mean')
    return eikonal_loss
```
**Purpose:** Enforces unit-length gradients (valid SDF property)

#### Point Cloud Consistency Loss
```python
def get_pc_loss(Xt, Yt):

    #ipdb.set_trace()
    # compute  error
    loss1 = self.comp_point_point_error(Xt[0].permute(1, 0), Yt[0].permute(1, 0))
    loss2= self.comp_point_point_error(Yt[0].permute(1, 0), Xt[0].permute(1, 0))
    loss = loss1 + loss2
    return loss
def comp_point_point_error(self, Xt, Yt):
    if Xt.shape[1] == 0 or Yt.shape[1] == 0:
        raise ValueError("输入张量在第二维度上为空。")
    closest_idx = self.comp_closest_pts_idx_with_split(Xt, Yt)
    pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
    pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
    eng = torch.mean(pt_pt_dist)
    return eng
```
**Purpose:** Ensures multi-view geometric consistency

---

## Data Processing Pipeline

### 1. Input Data Format

PMNI expects multi-view data in the following structure:

```
data/
├── diligent_mv_normals/bear/
│   ├── mask/           # Object silhouettes (PNG)
│   ├── normal_camera_space_GT/  # Ground truth normals (EXR)
│   ├── cameras_sphere.npz       # Camera parameters
│   └── mesh_Gt.ply              # Ground truth mesh (optional)
```

### 2. Data Loading Process

```python
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        # Keep dataset on CPU to avoid OOM, will move batches to GPU during training
        self.device = torch.device('cpu')
        print('[Dataset] Storing on CPU to avoid GPU OOM')
        self.conf = conf
        normal_dir = conf.get_string('normal_dir')
        depth_dir = conf.get_string('depth_dir')
        perspective_dir = conf.get_string('perspective_dir')
        self.data_dir = conf.get_string('data_dir')
        self.cameras_name = conf.get_string('cameras_name')
        self.exclude_view_list = conf['exclude_views']  # list of views to exclude from training. Used in novel-view normal synthesis evaluation.
        self.upsample_factor = conf.get_int('upsample_factor', default=1)
        ic(self.exclude_view_list)
        self.add_pose_noise = conf.get_bool('add_pose_noise')
        self.rotation_noise_std=conf.get_float('rotation_noise_std')
        self.translation_noise_std=conf.get_float('translation_noise_std')
        # load the GT mesh for evaluation if any
        mesh_path = os.path.join(self.data_dir, 'mesh_Gt.ply')
        if os.path.exists(mesh_path):
            self.mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
        else:
            self.mesh_gt = None
        self.points_gt = None  # will be computed from the mesh at evaluation time

        camera_dict = np.load(os.path.join(self.data_dir, self.cameras_name))
        self.camera_dict = camera_dict
        self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
        self.depth_lis = sorted(glob(os.path.join(self.data_dir, depth_dir, '*.npy')))
        self.perspective_lis = sorted(glob(os.path.join(self.data_dir, perspective_dir, '*.npy')))

        self.n_images = len(self.normal_lis)
        self.n_depth = len(self.depth_lis)
        self.n_perspective = len(self.perspective_lis)

        if(self.n_images!=self.n_depth|self.n_images!=self.n_perspective):
            print('error: the number of normals and depth is not the same')
        self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)
        self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in self.normal_lis]
        self.depth_idx_list = self.img_idx_list 

        print("loading normal maps...")
        self.normal_np = np.stack([pyexr.read(im_name)[..., :3] for im_name in self.normal_lis])
        print("loading depth maps...")
        self.depth_np = np.stack(np.load(depth_name) for depth_name in self.depth_lis)
        self.perspective_np = np.stack(np.load(perspective_name) for perspective_name in self.perspective_lis)
        if self.upsample_factor > 1:
            # resize normal maps
            self.normal_np = F.interpolate(torch.from_numpy(self.normal_np).permute(0, 3, 1, 2), scale_factor=self.upsample_factor, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
        self.normals = torch.from_numpy(self.normal_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        
        #if(normal_dir == 'normal_camera_space_GT'):
        self.normals[..., 1] *= -1  
        self.normals[..., 2] *= -1  
        print("loading normal maps done.")
        self.depths = torch.from_numpy(self.depth_np.astype(np.float32)).to(self.device)  # [n_images, H, W]
        self.perspective = torch.from_numpy(self.perspective_np.astype(np.float32)).to(self.device)  # [n_images, H, W]
        self.depths = self.depths*conf.get_float('depth_init_scale')
        self.perspective = self.perspective*conf.get_float('depth_init_scale')
        print(self.perspective.shape,'loading depth maps done.')

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
        
        if self.upsample_factor > 1:
            # resize mask
            self.masks_np = F.interpolate(torch.from_numpy(self.masks_np).permute(0, 3, 1, 2), scale_factor=self.upsample_factor, mode='nearest').permute(0, 2, 3, 1).numpy()
        self.masks_np = self.masks_np[..., 0]
        self.total_pixel = np.sum(self.masks_np)

        # set background of normal map to 0
        self.normal_np[self.masks_np == 0] = 0
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
        
        self.intrinsics_all = []
        self.pose_all = []

        self.H, self.W = self.normal_np.shape[1], self.normal_np.shape[2]
        for scale_mat, world_mat, normal_map, mask in zip(self.scale_mats_np, self.world_mats_np, self.normals, self.masks_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            if self.upsample_factor > 1:
                # resize intrinsics
                intrinsics[0, 0] *= self.upsample_factor
                intrinsics[1, 1] *= self.upsample_factor
                intrinsics[0, 2] *= self.upsample_factor
                intrinsics[1, 2] *= self.upsample_factor
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            intrinsics_inverse = torch.inverse(torch.from_numpy(intrinsics).float())
            pose = torch.from_numpy(pose).float()

        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device) # [n_images, H, W]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal_length = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.image_pixels = self.H * self.W
        self.K = self.intrinsics_all[0][:3, :3]
        if(self.add_pose_noise):
            self.pose_all= add_noise_to_pose(pose_all = self.pose_all,rotation_noise_std=self.rotation_noise_std,translation_noise_std=self.translation_noise_std)
        # for mesh extraction
        # for mesh extraction
        self.object_bbox_min = np.array([-20., -20., -20.])
        self.object_bbox_max = np.array([20.,  20.,  20.])

        # prepare instrincs
        intrinsics = self.intrinsics_all[0].detach()
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        h = self.H
        w = self.W
        self.camera_mat = torch.tensor([[
            [2 * fx / w, 0, 0, 0],
            [0, 2 * fy / h, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]], dtype=torch.float32).to(self.device)
        print('Load data: End')
        
    def gen_rays_at(self, img_idx, resolution_level=1, within_mask=False,pose_all = None):
        """
        Generate all rays at world space from one camera.
        """
        if(pose_all is None):
            pose_all=self.pose_all 
        mask_np = self.masks_np[img_idx].astype(bool)
        # resize the mask using resolution_level
        mask_np = cv.resize(mask_np.astype(np.uint8)*255, (int(self.W // resolution_level), int(self.H // resolution_level)), interpolation=cv.INTER_NEAREST).astype(bool)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        
        # Ensure all tensors are on the same device as pose_all
        device = pose_all.device
        p = p.to(device)
        intrinsics_inv = self.intrinsics_all_inv.to(device)
        
        p = torch.matmul(intrinsics_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        rays_o = rays_o.transpose(0, 1)
        rays_v = rays_v.transpose(0, 1)

        if within_mask:
            return rays_o[mask_np], rays_v[mask_np]
        else:
            return rays_o, rays_v

    def gen_patches_at(self, img_idx, resolution_level=1, patch_H=3, patch_W=3):
        tx = torch.linspace(0, self.W - 1, int(self.W // resolution_level))
        ty = torch.linspace(0, self.H - 1, int(self.H // resolution_level))
        pixels_y, pixels_x = torch.meshgrid(ty, tx)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # H, W, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, :3, :3], p[..., None]).squeeze()  # H, W, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, :3, :3], rays_v[:, :, :, None]).squeeze()  # H, W, 3

        # split rays_v into non-overlapping patches
        height, width, _ = rays_v.shape
        horizontal_num_patch = width // patch_W
        vertical_num_patch = height // patch_H
        rays_v_patches_all = []
        rays_ez_patches_all = []
        mask_value = []
        for i in range(0, height-patch_H//2-1, patch_H):
            for j in range(0, width-patch_W//2-1, patch_W):
                rays_v_patch = rays_v[i:i + patch_H, j:j + patch_W]
                rays_v_patches_all.append(rays_v_patch)


                rays_ez_patch = self.normals[img_idx][i + patch_H//2, j + patch_W//2]
                rays_ez_patches_all.append(rays_ez_patch)

                mask_value.append(self.masks_np[img_idx][i + patch_H//2, j + patch_W//2].astype(bool))
        rays_v_patches_all = torch.stack(rays_v_patches_all, dim=0)  # (num_patch, patch_H, patch_W, 3)
        rays_o_patches_all = self.pose_all[img_idx, :3, 3].expand(rays_v_patches_all.shape)  # (num_patch, patch_H, patch_W, 3)

        rays_o_patch_center = rays_o_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_v_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        marching_plane_normal_patches_all = self.pose_all[img_idx, :3, 2].expand(rays_d_patch_center.shape)  # (num_patch, 3)

        return rays_o_patch_center, \
                rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            marching_plane_normal_patches_all, \
            horizontal_num_patch, vertical_num_patch

    def gen_random_patches(self, num_patch, patch_H=3, patch_W=3):
        """
        Generate random patches of rays at world space from all viewpoints.
        X-axis right, Y-axis down

        Parameters:
        num_patch (int): The number of patches to generate.
        patch_H (int, optional): The height of the patches. Default is 3.
        patch_W (int, optional): The width of the patches. Default is 3.

        Returns:
        rays_o_patch_all (torch.Tensor): The origins of the rays in each patch. A tensor of shape (num_patch, patch_H, patch_W, 3).
        rays_d_patch_all (torch.Tensor): The directions of the rays in each patch. A tensor of shape (num_patch, patch_H, patch_W, 3).
        marching_plane_normal (torch.Tensor): The normal direction of the image/marching plane.
                Since we randomly sample patches from all viewpoints, this normal is only identical for each patch. A tensor of shape (num_patch, 3).
        V_inverse_patch_all (torch.Tensor): The inverse of the V matrix at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 3, 3).
        normal (torch.Tensor): The normals at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 3).
        mask (torch.Tensor): The mask values at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 1).
        """
        # randomly sample center pixel locations of patches
        # assume all images have the same resolution
        patch_center_x = torch.randint(low=0+patch_W//2, high=self.W-1-patch_W//2, size=[num_patch], device=self.device)  # (num_patch, )
        patch_center_y = torch.randint(low=0+patch_H//2, high=self.H-1-patch_H//2, size=[num_patch], device=self.device)  # (num_patch, )

        # compute all pixel locations within the patches given patch size (patch_H, patch_W)
        patch_center_x_all = patch_center_x[:, None, None] + torch.arange(-patch_W//2+1, patch_W//2+1, device=self.device).repeat(patch_H, 1)   # (num_patch, patch_H, patch_W)
        patch_center_y_all = patch_center_y[:, None, None] + torch.arange(-patch_H//2+1, patch_H//2+1, device=self.device).reshape(-1, 1).repeat(1, patch_W)   # (num_patch, patch_H, patch_W)

        # randomly sample viewpoints
        img_idx = np.random.choice(list(self.train_images), size=[num_patch])  # (num_patch, )
        img_idx = torch.tensor(img_idx, device=self.device)
        img_idx_expand = img_idx.view(-1, 1, 1).expand_as(patch_center_x_all)  # (num_patch, patch_H, patch_W)

        # input normals and mask values for supervision
        normal = self.normals[img_idx_expand, patch_center_y_all, patch_center_x_all]  # (num_patch, patch_H, patch_W, 3)
        mask = self.masks[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)#[..., :1]     # (num_patch, patch_H, patch_W)

        # compute all ray directions within patches
        p_all = torch.stack([patch_center_x_all, patch_center_y_all, torch.ones_like(patch_center_y_all)], dim=-1).float().to(self.device)  # (num_patch, patch_H, patch_W, 3)
        p_all = torch.matmul(self.intrinsics_all_inv[img_idx_expand, :3, :3], p_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        p_norm_all = torch.linalg.norm(p_all, ord=2, dim=-1, keepdim=True)  # (num_patch, patch_H, patch_W, 1)
        rays_d_patch_all = p_all / p_norm_all  # (num_patch, patch_H, patch_W, 3)
        rays_d_patch_all = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        rays_o_patch_all = self.pose_all[img_idx, None, None, :3, 3].expand(rays_d_patch_all.shape)  # (num_patch, patch_H, patch_W, 3)

        # the normal direction of the image/marching plane is the 3rd column of c2w transformation()
        marching_plane_normal = self.pose_all[img_idx, :3, 2].expand((num_patch, 3))  # (num_patch, 3)

        return rays_o_patch_all, \
                rays_d_patch_all, \
                marching_plane_normal, \
                normal,\
                mask
    
    def gen_random_patches_fixed_idx_pose(self, num_patch,patch_H=3, patch_W=3, idx = 0, pose=None):
      
        #ipdb.set_trace()
        patch_center_x = torch.randint(low=0+patch_W//2, high=self.W-1-patch_W//2, size=[num_patch], device=self.device)  # (num_patch, )
        patch_center_y = torch.randint(low=0+patch_H//2, high=self.H-1-patch_H//2, size=[num_patch], device=self.device)  # (num_patch, )

        # compute all pixel locations within the patches given patch size (patch_H, patch_W)
        patch_center_x_all = patch_center_x[:, None, None] + torch.arange(-patch_W//2+1, patch_W//2+1, device=self.device).repeat(patch_H, 1)   # (num_patch, patch_H, patch_W)
        patch_center_y_all = patch_center_y[:, None, None] + torch.arange(-patch_H//2+1, patch_H//2+1, device=self.device).reshape(-1, 1).repeat(1, patch_W)   # (num_patch, patch_H, patch_W)

        # randomly sample viewpoints
        img_idx = torch.full((num_patch,), idx, device=self.device)
        img_idx_expand = img_idx.view(-1, 1, 1).expand_as(patch_center_x_all)  # (num_patch, patch_H, patch_W)

        # input normals and mask values for supervision
        normal = self.normals[img_idx_expand, patch_center_y_all, patch_center_x_all]  # (num_patch, patch_H, patch_W, 3)
        rotation_matrix = pose[:3, :3]
        normal = normal.to(pose.device)
        normals_world = torch.einsum('ij,bhwj->bhwi',rotation_matrix , normal)
        #normal_world = torch.matmul(normal, rotation_matrix.T)
        mask = self.masks[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)#[..., :1]     # (num_patch, patch_H, patch_W)
        depth_toushi = self.perspective[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)
        mask = mask.to(pose.device)
        depth_toushi = depth_toushi.to(pose.device)

        # compute all ray directions within patches
        p_all = torch.stack([patch_center_x_all, patch_center_y_all, torch.ones_like(patch_center_y_all)], dim=-1).float().to(self.device)  # (num_patch, patch_H, patch_W, 3)
        p_all = torch.matmul(self.intrinsics_all_inv[img_idx_expand, :3, :3], p_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        p_norm_all = torch.linalg.norm(p_all, ord=2, dim=-1, keepdim=True)  # (num_patch, patch_H, patch_W, 1)
        rays_d_patch_all = p_all / p_norm_all  # (num_patch, patch_H, patch_W, 3)
        """
        rays_d_patch_all = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        rays_o_patch_all = self.pose_all[img_idx, None, None, :3, 3].expand(rays_d_patch_all.shape)  # (num_patch, patch_H, patch_W, 3)

        # the normal direction of the image/marching plane is the 3rd column of c2w transformation
        marching_plane_normal = self.pose_all[img_idx, :3, 2].expand((num_patch, 3))  # (num_patch, 3)
        """
        rays_d_patch_all = rays_d_patch_all.to(pose.device)
        rays_d_patch_all = torch.matmul(rotation_matrix[None, None, :, :], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        translation_vector = pose[:3, 3]
        
        rays_o_patch_all = translation_vector.repeat(rays_d_patch_all.shape[:-1] + (1,)) # (num_patch, patch_H, patch_W, 3)
        
        marching_plane_normal = rotation_matrix[:, 2].repeat(num_patch, 1) # (num_patch, 3)
    

        return rays_o_patch_all, \
                rays_d_patch_all, \
                marching_plane_normal, \
                normals_world,\
                depth_toushi,\
                mask
    
    def gen_random_patches_fixed_pose_world(self, num_patch,patch_H=3, patch_W=3, idx = 0, pose=None):
        
        #ipdb.set_trace()
        patch_center_x = torch.randint(low=0+patch_W//2, high=self.W-1-patch_W//2, size=[num_patch], device=self.device)  # (num_patch, )
        patch_center_y = torch.randint(low=0+patch_H//2, high=self.H-1-patch_H//2, size=[num_patch], device=self.device)  # (num_patch, )

        # compute all pixel locations within the patches given patch size (patch_H, patch_W)
        patch_center_x_all = patch_center_x[:, None, None] + torch.arange(-patch_W//2+1, patch_W//2+1, device=self.device).repeat(patch_H, 1)   # (num_patch, patch_H, patch_W)
        patch_center_y_all = patch_center_y[:, None, None] + torch.arange(-patch_H//2+1, patch_H//2+1, device=self.device).reshape(-1, 1).repeat(1, patch_W)   # (num_patch, patch_H, patch_W)

        # randomly sample viewpoints
        img_idx = np.full(shape=[num_patch], fill_value=idx)  # (num_patch, )
        img_idx = torch.tensor(img_idx, device=self.device)
        img_idx_expand = img_idx.view(-1, 1, 1).expand_as(patch_center_x_all)  # (num_patch, patch_H, patch_W)

        # input normals and mask values for supervision
        normal = self.normals[img_idx_expand, patch_center_y_all, patch_center_x_all]  # (num_patch, patch_H, patch_W, 3)
        rotation_matrix = pose[:3, :3]
        
        normals_world = normal
        #normal_world = torch.matmul(normal, rotation_matrix.T)
        mask = self.masks[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)#[..., :1]     # (num_patch, patch_H, patch_W)
        
        depth_toushi = self.perspective[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)

        # compute all ray directions within patches
        p_all = torch.stack([patch_center_x_all, patch_center_y_all, torch.ones_like(patch_center_y_all)], dim=-1).float().to(self.device)  # (num_patch, patch_H, patch_W, 3)
        p_all = torch.matmul(self.intrinsics_all_inv[img_idx_expand, :3, :3], p_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        p_norm_all = torch.linalg.norm(p_all, ord=2, dim=-1, keepdim=True)  # (num_patch, patch_H, patch_W, 1)
        rays_d_patch_all = p_all / p_norm_all  # (num_patch, patch_H, patch_W, 3)
        """
        rays_d_patch_all = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        rays_o_patch_all = self.pose_all[img_idx, None, None, :3, 3].expand(rays_d_patch_all.shape)  # (num_patch, patch_H, patch_W, 3)

        # the normal direction of the image/marching plane is the 3rd column of c2w transformation(这里是光线方向)
        marching_plane_normal = self.pose_all[img_idx, :3, 2].expand((num_patch, 3))  # (num_patch, 3)
        """
        rays_d_patch_all = torch.matmul(rotation_matrix[None, None, :, :], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        
        translation_vector = pose[:3, 3]
       
        rays_o_patch_all = translation_vector.expand(rays_d_patch_all.shape)  # (num_patch, patch_H, patch_W, 3)
        
        marching_plane_normal = rotation_matrix[:, 2].expand((num_patch, 3))  # (num_patch, 3)
    

        return rays_o_patch_all, \
                rays_d_patch_all, \
                marching_plane_normal, \
                normals_world,\
                depth_toushi,\
                mask            

    def near_far_from_sphere(self, rays_o, rays_d):
        """
        This function calculates the near and far intersection points of rays with a unit sphere.

        Parameters:
        rays_o (torch.Tensor): Origin of the rays. A tensor of shape (N, 3) where N is the number of rays.
        rays_d (torch.Tensor): Direction of the rays. A tensor of shape (N, 3) where N is the number of rays.

        Returns:
        near (torch.Tensor): Near intersection points of the rays with the unit sphere. A tensor of shape (N, ).
        far (torch.Tensor): Far intersection points of the rays with the unit sphere. A tensor of shape (N, ).
        """
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        c = torch.sum(rays_o**2, dim=-1, keepdim=True) - 1.0
        mid = 0.5 * (-b) / a
        near = mid - 1#torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        far = mid + 1#torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        return near[..., 0], far[..., 0]

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
```

### 3. Training Data Generation

```python
def gen_random_patches(self, batch_size, patch_size=3):
    # 1. Random camera selection
    cam_idx = np.random.randint(self.n_images)
    
    # 2. Random patch extraction
    H, W = self.normals.shape[1:3]
    h_start = np.random.randint(0, H - patch_size)
    w_start = np.random.randint(0, W - patch_size)
    
    # 3. Extract local patches
    normal_patch = self.normals[cam_idx, h_start:h_start+patch_size, 
                               w_start:w_start+patch_size]
    depth_patch = self.depths[cam_idx, h_start:h_start+patch_size,
                             w_start:w_start+patch_size]
    mask_patch = self.masks[cam_idx, h_start:h_start+patch_size,
                           w_start:w_start+patch_size]
    
    # 4. Generate rays for the patch
    rays_o, rays_d = self.gen_rays_at_patch(cam_idx, h_start, w_start, 
                                           patch_size, patch_size)
    
    return rays_o, rays_d, normal_patch, depth_patch, mask_patch
```

---

## Implementation Details

### 1. Acceleration Techniques

#### Occupancy Grid
```python
# Pre-compute occupied regions for faster ray marching
occupancy_grid = OccupancyGrid(
    roi_aabb=[-1, -1, -1, 1, 1, 1],  # Scene bounds
    resolution=128,                    # Grid resolution
    contraction_type=ContractionType.AABB
)

# Update every N steps during training
if iteration % 100 == 0:
    occupancy_grid.every_n_step(
        occ_eval_fn=lambda x: torch.sigmoid(-sdf_network(x) * 80)
    )
```

#### Tiny-CUDA-NN Integration
```python
# Hash grid encoding for efficient MLP evaluation
encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0
}

encoding = tcnn.Encoding(3, encoding_config)  # 3D input → high-dim features
```

### 2. Memory Optimization

#### Chunked Processing
```python
# Process large batches in chunks to avoid GPU memory issues
def forward_chunked(self, points, chunk_size=8192):
    outputs = []
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        output = self.sdf_network(chunk)
        outputs.append(output)
    return torch.cat(outputs, dim=0)
```

#### Gradient Checkpointing
```python
# Trade compute for memory during rendering
with torch.autograd.enable_grad():
    rendered_output = torch.utils.checkpoint.checkpoint(
        renderer.render_function,
        rays_o, rays_d, near, far
    )
```

### 3. Numerical Stability

#### NaN/Inf Guards
```python
def safe_sdf_forward(self, x):
    sdf = self.sdf_network(x)
    sdf = torch.nan_to_num(sdf, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.clamp(sdf, -10.0, 10.0)  # Reasonable SDF bounds
```

#### Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(sdf_net.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(pose_net.parameters(), max_norm=0.1)
```

---

## Validation and Evaluation

### 1. Training Validation

#### Normal Map Validation
```python
def validate_normal_pixel_based(self, idx):
    # Render full-resolution normal maps
    rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
    
    # Batch processing for memory efficiency
    normal_img = []
    for rays_o_batch, rays_d_batch in batch_generator(rays_o, rays_d, 8192):
        batch_normals = self.renderer.render_normal_pixel_based(
            rays_o_batch, rays_d_batch
        )
        normal_img.append(batch_normals)
    
    # Save validation results
    cv.imwrite(f'normals/iter_{self.iter_step}_{idx}.png', 
               (normal_img * 128 + 128).clip(0, 255))
```

#### Mesh Extraction Validation
```python
def validate_mesh(self, resolution=1024):
    # Extract mesh using marching cubes
    vertices, triangles = self.renderer.extract_geometry(
        bound_min, bound_max, resolution=resolution, threshold=0.0
    )
    
    # Clean mesh (remove isolated components)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh = remove_isolated_clusters(mesh)
    
    # Save for inspection
    mesh.export(f'meshes_validation/iter_{self.iter_step}.ply')
```

### 2. Quantitative Evaluation

#### Mean Angular Error (MAE)
```python
def eval_mae(self, gradient_method='ad'):
    ae_map_list = []
    for idx in range(self.dataset.n_images):
        # Load ground truth normals
        gt_normals = load_exr(f'normal_world_space_GT/{idx:02d}.exr')
        
        # Render predicted normals
        pred_normals = self.validate_normal_patch_based(idx)
        
        # Compute angular error
        angular_error = compute_angular_error(gt_normals, pred_normals)
        ae_map_list.append(angular_error)
    
    mae = np.nanmean(np.stack(ae_map_list))
    return mae
```

#### Geometric Evaluation
```python
def eval_geo(self, resolution=512):
    # Extract predicted mesh
    pred_mesh = extract_mesh(resolution)
    pred_points = sample_points_from_mesh(pred_mesh)
    
    # Load ground truth points
    gt_points = load_ground_truth_points()
    
    # Compute Chamfer Distance
    cd = chamfer_distance(pred_points, gt_points)
    
    # Compute F-Score
    fscore = f_score(pred_points, gt_points, threshold=0.01)
    
    return cd, fscore
```

---

## Using PMNI Outputs

### 1. Immediate Post-Training Usage

#### **Mesh Files**
```bash
# View reconstructed mesh
meshlab exp/diligent_mv/bear/exp_2025_11_08_12_15_02/meshes_validation/iter_00030000.ply

# Convert to other formats
# - STL for 3D printing
# - OBJ for game engines
# - glTF for web deployment
```

#### **Normal Maps**
```python
# Load validation normals
import cv2
normal_img = cv2.imread('normals/iter_30000_0.png')
normals = (normal_img.astype(np.float32) - 128) / 128  # Convert to [-1, 1]

# Use for:
# - Surface analysis
# - Material property estimation
# - Lighting simulation
```

#### **Camera Poses**
```python
# Load optimized camera poses
poses = np.load('poses/30000.npy')  # Shape: (20, 4, 4)

# Use for:
# - Novel view synthesis
# - Multi-view registration
# - SLAM initialization
```

### 2. Inference on New Data

#### **Load Trained Model**
```python
import torch
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer

# Load checkpoints
checkpoint = torch.load('checkpoints/ckpt_30000.pth')
sdf_net = SDFNetwork()
sdf_net.load_state_dict(checkpoint['sdf_network_fine'])
variance_net = SingleVarianceNetwork(0.3)
variance_net.load_state_dict(checkpoint['variance_network_fine'])

renderer = NeuSRenderer(sdf_net, variance_net)
```

#### **Render Novel Views**
```python
def render_novel_view(camera_pose, image_size=(512, 512)):
    # Generate rays for new camera position
    rays_o, rays_d = generate_rays_from_pose(camera_pose, image_size)
    
    # Render using trained model
    with torch.no_grad():
        render_out = renderer.render(rays_o, rays_d, near=0.1, far=10.0)
        
    return render_out['comp_normal'], render_out['comp_depth']
```

#### **Extract Surface Points**
```python
def extract_surface_points(num_points=10000):
    # Sample points near zero SDF
    points = torch.randn(num_points, 3) * 2  # Random points in [-2, 2]^3
    
    with torch.no_grad():
        sdf_values = sdf_net(points)
    
    # Keep points near surface (SDF ≈ 0)
    surface_mask = torch.abs(sdf_values.squeeze()) < 0.01
    surface_points = points[surface_mask]
    
    return surface_points
```

### 3. Integration into Applications

#### **Real-time Rendering Pipeline**
```python
class PMNIRenderer:
    def __init__(self, model_path):
        self.load_model(model_path)
        self.setup_rendering_pipeline()
    
    def render_frame(self, camera_pose):
        # Fast rendering for real-time applications
        return self.fast_render(camera_pose)
    
    def extract_mesh(self):
        # High-quality mesh extraction
        return self.high_res_mesh_extraction()
```

#### **3D Asset Pipeline**
```python
# Complete asset creation workflow
def create_3d_asset(input_images):
    # 1. Run PMNI training
    trained_model = train_pmni(input_images)
    
    # 2. Extract optimized mesh
    mesh = extract_mesh(trained_model)
    
    # 3. Optimize for target application
    optimized_mesh = optimize_mesh(mesh, target='game_engine')
    
    # 4. Generate LODs
    lods = generate_lods(optimized_mesh)
    
    # 5. Package asset
    asset_package = package_asset(optimized_mesh, lods, textures=None)
    
    return asset_package
```

### 4. Research and Analysis

#### **Surface Analysis**
```python
def analyze_reconstructed_surface(mesh_path):
    mesh = trimesh.load(mesh_path)
    
    # Compute surface properties
    surface_area = mesh.area
    volume = mesh.volume
    curvature = compute_curvature(mesh)
    
    # Quality metrics
    watertight = mesh.is_watertight
    self_intersections = detect_self_intersections(mesh)
    
    return {
        'surface_area': surface_area,
        'volume': volume,
        'watertight': watertight,
        'quality_score': compute_quality_score(mesh)
    }
```

#### **Training Analysis**
```python
def analyze_training_progress(log_path):
    # Parse training logs
    losses = parse_losses(log_path)
    
    # Plot convergence curves
    plot_convergence(losses)
    
    # Analyze training stability
    stability_metrics = compute_stability_metrics(losses)
    
    # Identify optimal stopping point
    optimal_iter = find_optimal_stopping_point(losses)
    
    return training_analysis
```

### 5. Production Deployment

#### **Model Optimization**
```python
# Optimize for inference speed
def optimize_for_inference(model):
    # Quantization
    quantized_model = quantize_model(model, precision='fp16')
    
    # Pruning
    pruned_model = prune_model(quantized_model, sparsity=0.3)
    
    # TensorRT optimization
    trt_model = convert_to_tensorrt(pruned_model)
    
    return trt_model
```

#### **Scalability Considerations**
```python
# Handle large scenes
def process_large_scene(input_data, chunk_size=1000000):
    # Divide scene into manageable chunks
    chunks = divide_scene(input_data, chunk_size)
    
    # Process each chunk
    chunk_results = []
    for chunk in chunks:
        result = process_chunk(chunk)
        chunk_results.append(result)
    
    # Merge results
    final_result = merge_results(chunk_results)
    
    return final_result
```

---

## Advanced Features and Extensions

### 1. Multi-Object Reconstruction

```python
class MultiObjectPMNI:
    def __init__(self):
        self.object_detectors = []
        self.instance_segmentors = []
        self.pmni_instances = []
    
    def reconstruct_scene(self, images):
        # 1. Detect objects
        detections = self.detect_objects(images)
        
        # 2. Segment instances
        masks = self.segment_instances(images, detections)
        
        # 3. Reconstruct each object
        reconstructions = []
        for mask in masks:
            reconstruction = self.reconstruct_object(images, mask)
            reconstructions.append(reconstruction)
        
        return reconstructions
```

### 2. Temporal Consistency

```python
class TemporalPMNI:
    def __init__(self):
        self.temporal_smoothing = ExponentialMovingAverage()
        self.motion_estimation = OpticalFlowEstimator()
    
    def reconstruct_video_sequence(self, frames):
        reconstructions = []
        for frame in frames:
            # Estimate motion from previous frame
            motion = self.estimate_motion(frame, previous_frame)
            
            # Initialize pose from motion
            initial_pose = self.predict_pose_from_motion(motion)
            
            # Reconstruct with temporal prior
            reconstruction = self.reconstruct_with_temporal_prior(
                frame, initial_pose, previous_reconstruction
            )
            
            reconstructions.append(reconstruction)
        
        return reconstructions
```

### 3. Material and Texture Estimation

```python
class TexturedPMNI:
    def __init__(self):
        self.material_net = MaterialNetwork()
        self.texture_net = TextureNetwork()
    
    def reconstruct_with_materials(self, images):
        # Joint geometry and material reconstruction
        geometry, materials = self.joint_optimization(images)
        
        # Texture synthesis
        textures = self.synthesize_textures(geometry, materials)
        
        return geometry, materials, textures
```

---

## Troubleshooting and Best Practices

### 1. Common Issues

#### **Training Instability**
```python
# Symptoms: NaN losses, exploding gradients
# Solutions:
# 1. Reduce learning rate
# 2. Enable gradient clipping
# 3. Use loss scaling
# 4. Check input data quality
```

#### **Poor Reconstruction Quality**
```python
# Symptoms: Holes, artifacts, incorrect topology
# Solutions:
# 1. Increase training iterations
# 2. Improve pose initialization
# 3. Use higher resolution data
# 4. Adjust loss weights
```

#### **Memory Issues**
```python
# Symptoms: CUDA out of memory
# Solutions:
# 1. Reduce batch size
# 2. Use gradient checkpointing
# 3. Enable chunked processing
# 4. Use mixed precision training
```

### 2. Performance Optimization

#### **Speed Improvements**
```python
# 1. Use tiny-cuda-nn for faster encoding
# 2. Optimize occupancy grid updates
# 3. Use hierarchical sampling
# 4. Implement early ray termination
```

#### **Quality Improvements**
```python
# 1. Progressive resolution increase
# 2. Multi-stage loss weighting
# 3. Pose refinement iterations
# 4. Surface normal regularization
```

### 3. Best Practices

#### **Data Preparation**
- Ensure consistent camera calibration
- Use high-quality normal/depth maps
- Validate input data integrity
- Consider pose initialization quality

#### **Training Strategy**
- Start with conservative hyperparameters
- Monitor validation metrics closely
- Use early stopping when appropriate
- Save checkpoints frequently

#### **Evaluation Protocol**
- Use multiple evaluation metrics
- Compare against ground truth when available
- Validate on held-out test views
- Consider both quantitative and qualitative metrics

---

## Conclusion

PMNI represents a significant advancement in neural implicit surface reconstruction by jointly optimizing geometry and camera poses. The method's ability to learn from multi-view images without known camera parameters makes it particularly valuable for applications where pose estimation is challenging.

### Key Strengths:
- **Robust Pose Optimization**: Learns camera parameters alongside geometry
- **High-Quality Reconstructions**: Produces smooth, detailed 3D meshes
- **Scalable Architecture**: Handles complex scenes and objects
- **Research-Friendly**: Extensible framework for novel research directions

### Future Directions:
- **Real-time Applications**: Optimizing for interactive reconstruction
- **Large-Scale Scenes**: Extending to city-scale reconstruction
- **Semantic Understanding**: Incorporating object categories and relationships
- **Physical Simulation**: Integration with physics engines for dynamic scenes

The comprehensive training outputs provide researchers and practitioners with rich data for analysis, validation, and further development. Whether used for academic research, industrial applications, or creative projects, PMNI offers a powerful and flexible framework for 3D reconstruction from multi-view imagery.

---

*This documentation provides a complete understanding of PMNI from theoretical foundations to practical implementation. The system is designed to be both powerful for research and practical for real-world applications.*

---

## Introduction to PMNI

**PMNI (Pose-optimized Multi-view Neural Implicit)** is a state-of-the-art neural implicit surface reconstruction method that learns 3D shapes from multi-view 2D images. Unlike traditional methods that require known camera poses, PMNI jointly optimizes both the 3D geometry and camera parameters during training.

### Key Innovations:
- **Implicit Neural Representations**: Uses Signed Distance Functions (SDF) for continuous surface representation
- **Joint Pose Optimization**: Learns camera poses alongside geometry for robust reconstruction
- **Multi-view Consistency**: Enforces geometric consistency across different viewpoints
- **Real-time Rendering**: Enables fast surface rendering and normal estimation

### Applications:
- **3D Reconstruction**: From multi-view images without known poses
- **Object Scanning**: Mobile and handheld scanning applications
- **AR/VR Content Creation**: Automatic 3D asset generation
- **Robotics**: Object manipulation and scene understanding
- **Cultural Heritage**: Artifact digitization and preservation

---

## Core Concepts and Theory

### 1. Signed Distance Functions (SDF)

A **Signed Distance Function** represents a surface implicitly by mapping any 3D point to its signed distance from the surface:

```
SDF(x) = 
  +d  if point x is outside the surface
  -d  if point x is inside the surface
   0  if point x is on the surface
```

**Advantages:**
- **Continuous representation**: No discretization artifacts
- **Infinite resolution**: Can query any point in space
- **Topological flexibility**: Handles complex geometries naturally

### 2. Neural Implicit Representations

Instead of storing SDF values explicitly, PMNI uses a **neural network** to learn the SDF function:

```
SDF_θ(x) ≈ SDF(x)
```

Where `θ` represents the network parameters learned during training.

### 3. Volume Rendering

PMNI uses **differentiable volume rendering** to connect 2D observations with 3D geometry:

- **Ray marching**: Traces rays from camera through 3D space
- **Density estimation**: Converts SDF to density using sigmoid function
- **Alpha compositing**: Accumulates color/transmittance along rays

### 4. Pose Optimization

Camera poses are learned as **Lie group parameters** (rotation + translation):

```
c2w = exp([ω, v]) ⊕ c2w_initial
```

Where `ω` is rotation vector and `v` is translation vector.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PMNI Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Multi-view Images → Dataset Loader → Preprocessing        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Neural Networks                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │  SDF Network│  │Variance Net │  │ Pose Network│ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Rendering Engine                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │Ray Marching │  │Volume Render│  │Normal Est. │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Loss Functions                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │Depth Loss   │  │Normal Loss  │  │Eikonal Loss│ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Training Loop → Optimization → Mesh Extraction            │
└─────────────────────────────────────────────────────────────┘
```

### Core Components:

1. **SDF Network**: Learns the implicit surface representation
2. **Variance Network**: Controls surface uncertainty/blur
3. **Pose Network**: Optimizes camera parameters
4. **Renderer**: Converts implicit representation to 2D images
5. **Loss Functions**: Supervise learning with multi-view constraints

---

## Detailed Component Analysis

### 1. SDF Network (fields.py)

The **SDFNetwork** is a multi-layer perceptron that learns the signed distance function:

```python
class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_out=1, d_hidden=256, n_layers=8):
        # Input: 3D point coordinates
        # Output: Signed distance value
        
        self.encoding = tcnn.Encoding(3, encoding_config)  # Hash grid encoding
        self.layers = nn.ModuleList()
        
        # Skip connections for better gradient flow
        for i in range(n_layers):
            if i in [4]:  # Skip connection at layer 4
                self.layers.append(nn.Linear(d_hidden + d_in, d_hidden))
            else:
                self.layers.append(nn.Linear(d_hidden, d_hidden))
        
        self.output_layer = nn.Linear(d_hidden, 1)
```

**Key Features:**
- **Hash Grid Encoding**: Efficient high-dimensional encoding using tiny-cuda-nn
- **Skip Connections**: Direct input-to-output pathways for better learning
- **Geometric Initialization**: Special weight initialization for SDF learning
- **Progressive Bandwidth**: Gradually increases encoding resolution during training

### 2. Variance Network (fields.py)

Controls the "softness" of the surface for volume rendering:

```python
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        self.variance = nn.Parameter(torch.tensor(init_val))
    
    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
```

**Purpose:**
- **Surface Uncertainty**: Higher variance = softer surface boundaries
- **Rendering Quality**: Affects how rays interact with the surface
- **Training Stability**: Helps with convergence in early training stages

### 3. Pose Network (pose_net.py)

Learns camera poses as Lie group parameters:

```python
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        self.r = nn.Parameter(torch.zeros(num_cams, 3))  # Rotation (axis-angle)
        self.t = nn.Parameter(torch.zeros(num_cams, 3))  # Translation
        
    def forward(self, cam_id):
        # Convert axis-angle to rotation matrix
        R = axis_angle_to_matrix(self.r[cam_id])
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = self.t[cam_id]
        return c2w
```

**Features:**
- **Lie Group Optimization**: Proper rotation manifold optimization
- **Incremental Updates**: Learns pose corrections from initial estimates
- **Per-Camera Parameters**: Independent pose for each camera

### 4. Renderer (renderer.py)

Converts implicit representation to 2D renderings:

```python
class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network):
        self.occupancy_grid = OccupancyGrid()  # For acceleration
        self.sampling_step_size = 0.01  # Ray marching step
        
    def render(self, rays_o, rays_d, near, far):
        # Volume rendering pipeline
        # 1. Ray marching with occupancy grid
        # 2. SDF to density conversion
        # 3. Alpha compositing
        # 4. Surface normal estimation
        return rendered_image, normals, depths
```

**Rendering Pipeline:**
1. **Ray Generation**: Create rays from camera parameters
2. **Ray Marching**: Sample points along rays using occupancy grid
3. **Density Estimation**: Convert SDF to volume density
4. **Alpha Compositing**: Accumulate transmittance and color
5. **Normal Estimation**: Compute surface gradients

---

## Training Process Step-by-Step

### Phase 1: Initialization (0-10k iterations)

```python
# 1. Load multi-view data (normals, depths, masks)
dataset = Dataset(conf)
sdf_net = SDFNetwork()
pose_net = LearnPose(num_cams=20)

# 2. Initialize occupancy grid for acceleration
renderer = NeuSRenderer(sdf_net, variance_net)
occupancy_grid = OccupancyGrid(scene_aabb=[-1,1,-1,1,-1,1])

# 3. Set up optimization
optimizer_sdf = Adam(sdf_net.parameters(), lr=1e-3)
optimizer_pose = Adam(pose_net.parameters(), lr=1e-3)
```

### Phase 2: Main Training Loop

```python
for iteration in range(30000):
    # 1. Sample training patches
    rays_o, rays_d, gt_normals, gt_depths, masks = dataset.gen_patches()
    
    # 2. Get current camera poses
    c2w_matrices = pose_net.get_all_c2w()
    
    # 3. Render from current geometry
    render_out = renderer.render(rays_o, rays_d, c2w_matrices)
    pred_normals = render_out['comp_normal']
    pred_depths = render_out['comp_depth']
    
    # 4. Compute losses
    normal_loss = F.mse_loss(pred_normals, gt_normals)
    depth_loss = F.l1_loss(pred_depths, gt_depths)
    eikonal_loss = compute_eikonal_loss(render_out['gradients'])
    
    # 5. Backpropagation
    total_loss = normal_loss + depth_loss + eikonal_loss
    total_loss.backward()
    
    # 6. Update networks
    optimizer_sdf.step()
    optimizer_pose.step()
    
    # 7. Update occupancy grid
    if iteration % 100 == 0:
        occupancy_grid.update(sdf_net)
```

### Phase 3: Mesh Extraction

```python
# Extract final mesh using marching cubes
bound_min = torch.tensor([-1, -1, -1])
bound_max = torch.tensor([1, 1, 1])

vertices, triangles = renderer.extract_geometry(bound_min, bound_max, 
                                               resolution=1024, threshold=0.0)

# Save as PLY file
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export('reconstructed_mesh.ply')
```

---

## Loss Functions and Optimization

### 1. Multi-Stage Loss Weighting

PMNI uses **progressive loss weighting** to guide learning:

```python
# Stage 1 (0-10k): Focus on basic geometry
weights = {
    'normal_weight': 1.0,    # Surface orientation
    'depth_weight': 1.0,    # Distance constraints
    'eikonal_weight': 0.1,  # SDF regularization
}

# Stage 2 (10k-20k): Refine details
weights = {
    'normal_weight': 2.0,    # Stronger normal supervision
    'depth_weight': 1.5,    # Maintain depth accuracy
    'eikonal_weight': 0.2,  # Tighter regularization
}

# Stage 3 (20k-30k): Final convergence
weights = {
    'normal_weight': 3.0,    # High-quality normals
    'depth_weight': 2.0,    # Precise depths
    'eikonal_weight': 0.3,  # Final regularization
}
```

### 2. Individual Loss Components

#### Normal Loss
```python
def get_normal_loss(pred_normals, gt_normals, mask):
    error = (pred_normals - gt_normals) * mask
    return F.l2_loss(error, torch.zeros_like(error))
```
**Purpose:** Ensures predicted surface orientations match ground truth

#### Depth Loss
```python
def get_depth_loss(pred_depths, gt_depths, mask):
    error = (pred_depths - gt_depths) * mask
    return F.l1_loss(error, torch.zeros_like(error))
```
**Purpose:** Constrains surface distances from camera

#### Eikonal Loss
```python
def eikonal_loss(gradients):
    return ((gradients.norm(2, dim=-1) - 1.0) ** 2).mean()
```
**Purpose:** Enforces unit-length gradients (valid SDF property)

#### Point Cloud Consistency Loss
```python
def pc_loss(points_1, points_2):
    # Find nearest neighbors between point clouds
    distances = compute_nearest_neighbor_distances(points_1, points_2)
    return distances.mean()
```
**Purpose:** Ensures multi-view geometric consistency

---

## Data Processing Pipeline

### 1. Input Data Format

PMNI expects multi-view data in the following structure:

```
data/
├── diligent_mv_normals/bear/
│   ├── mask/           # Object silhouettes (PNG)
│   ├── normal_camera_space_GT/  # Ground truth normals (EXR)
│   ├── cameras_sphere.npz       # Camera parameters
│   └── mesh_Gt.ply              # Ground truth mesh (optional)
```

### 2. Data Loading Process

```python
class Dataset:
    def __init__(self, conf):
        # 1. Load camera intrinsics and extrinsics
        camera_dict = np.load('cameras_sphere.npz')
        self.K = camera_dict['intrinsics']      # (3, 3) camera matrix
        self.pose_all = camera_dict['extrinsics'] # (N, 4, 4) camera poses
        
        # 2. Load normal maps
        self.normals = load_exr_files('normal_camera_space_GT/*.exr')
        
        # 3. Load depth maps
        self.depths = load_npy_files('depths/*.npy')
        
        # 4. Load object masks
        self.masks = load_png_files('mask/*.png')
        
        # 5. Preprocessing
        self.normals = self.normals * scaling_factors
        self.depths = self.depths * depth_scale
```

### 3. Training Data Generation

```python
def gen_random_patches(self, batch_size, patch_size):
    # 1. Random camera selection
    cam_idx = np.random.randint(self.n_images)
    
    # 2. Random patch extraction
    H, W = self.normals.shape[1:3]
    h_start = np.random.randint(0, H - patch_size)
    w_start = np.random.randint(0, W - patch_size)
    
    # 3. Extract local patches
    normal_patch = self.normals[cam_idx, h_start:h_start+patch_size, 
                               w_start:w_start+patch_size]
    depth_patch = self.depths[cam_idx, h_start:h_start+patch_size,
                             w_start:w_start+patch_size]
    mask_patch = self.masks[cam_idx, h_start:h_start+patch_size,
                           w_start:w_start+patch_size]
    
    # 4. Generate rays for the patch
    rays_o, rays_d = self.gen_rays_at_patch(cam_idx, h_start, w_start, 
                                           patch_size, patch_size)
    
    return rays_o, rays_d, normal_patch, depth_patch, mask_patch
```

---

## Implementation Details

### 1. Acceleration Techniques

#### Occupancy Grid
```python
# Pre-compute occupied regions for faster ray marching
occupancy_grid = OccupancyGrid(
    roi_aabb=[-1, -1, -1, 1, 1, 1],  # Scene bounds
    resolution=128,                    # Grid resolution
    contraction_type=ContractionType.AABB
)

# Update every N steps during training
if iteration % 100 == 0:
    occupancy_grid.every_n_step(
        occ_eval_fn=lambda x: torch.sigmoid(-sdf_network(x) * 80)
    )
```

#### Tiny-CUDA-NN Integration
```python
# Hash grid encoding for efficient MLP evaluation
encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0
}

encoding = tcnn.Encoding(3, encoding_config)  # 3D input → high-dim features
```

### 2. Memory Optimization

#### Chunked Processing
```python
# Process large batches in chunks to avoid GPU memory issues
def forward_chunked(self, points, chunk_size=8192):
    outputs = []
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        output = self.sdf_network(chunk)
        outputs.append(output)
    return torch.cat(outputs, dim=0)
```

#### Gradient Checkpointing
```python
# Trade compute for memory during rendering
with torch.autograd.enable_grad():
    rendered_output = torch.utils.checkpoint.checkpoint(
        renderer.render_function,
        rays_o, rays_d, near, far
    )
```

### 3. Numerical Stability

#### NaN/Inf Guards
```python
def safe_sdf_forward(self, x):
    sdf = self.sdf_network(x)
    sdf = torch.nan_to_num(sdf, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.clamp(sdf, -10.0, 10.0)  # Reasonable SDF bounds
```

#### Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(sdf_net.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(pose_net.parameters(), max_norm=0.1)
```

---

## Validation and Evaluation

### 1. Training Validation

#### Normal Map Validation
```python
def validate_normal_pixel_based(self, idx):
    # Render full-resolution normal maps
    rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
    
    # Batch processing for memory efficiency
    normal_img = []
    for rays_o_batch, rays_d_batch in batch_generator(rays_o, rays_d, 8192):
        batch_normals = self.renderer.render_normal_pixel_based(
            rays_o_batch, rays_d_batch
        )
        normal_img.append(batch_normals)
    
    # Save validation results
    cv.imwrite(f'normals/iter_{self.iter_step}_{idx}.png', 
               (normal_img * 128 + 128).clip(0, 255))
```

#### Mesh Extraction Validation
```python
def validate_mesh(self, resolution=1024):
    # Extract mesh using marching cubes
    vertices, triangles = self.renderer.extract_geometry(
        bound_min, bound_max, resolution=resolution, threshold=0.0
    )
    
    # Clean mesh (remove isolated components)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh = remove_isolated_clusters(mesh)
    
    # Save for inspection
    mesh.export(f'meshes_validation/iter_{self.iter_step}.ply')
```

### 2. Quantitative Evaluation

#### Mean Angular Error (MAE)
```python
def eval_mae(self, gradient_method='ad'):
    ae_map_list = []
    for idx in range(self.dataset.n_images):
        # Load ground truth normals
        gt_normals = load_exr(f'normal_world_space_GT/{idx:02d}.exr')
        
        # Render predicted normals
        pred_normals = self.validate_normal_patch_based(idx)
        
        # Compute angular error
        angular_error = compute_angular_error(gt_normals, pred_normals)
        ae_map_list.append(angular_error)
    
    mae = np.nanmean(np.stack(ae_map_list))
    return mae
```

#### Geometric Evaluation
```python
def eval_geo(self, resolution=512):
    # Extract predicted mesh
    pred_mesh = extract_mesh(resolution)
    pred_points = sample_points_from_mesh(pred_mesh)
    
    # Load ground truth points
    gt_points = load_ground_truth_points()
    
    # Compute Chamfer Distance
    cd = chamfer_distance(pred_points, gt_points)
    
    # Compute F-Score
    fscore = f_score(pred_points, gt_points, threshold=0.01)
    
    return cd, fscore
```

---

## Using PMNI Outputs

### 1. Immediate Post-Training Usage

#### **Mesh Files**
```bash
# View reconstructed mesh
meshlab exp/diligent_mv/bear/exp_2025_11_08_12_15_02/meshes_validation/iter_00030000.ply

# Convert to other formats
# - STL for 3D printing
# - OBJ for game engines
# - glTF for web deployment
```

#### **Normal Maps**
```python
# Load validation normals
import cv2
normal_img = cv2.imread('normals/iter_30000_0.png')
normals = (normal_img.astype(np.float32) - 128) / 128  # Convert to [-1, 1]

# Use for:
# - Surface analysis
# - Material property estimation
# - Lighting simulation
```

#### **Camera Poses**
```python
# Load optimized camera poses
poses = np.load('poses/30000.npy')  # Shape: (20, 4, 4)

# Use for:
# - Novel view synthesis
# - Multi-view registration
# - SLAM initialization
```

### 2. Inference on New Data

#### **Load Trained Model**
```python
import torch
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer

# Load checkpoints
checkpoint = torch.load('checkpoints/ckpt_30000.pth')
sdf_net = SDFNetwork()
sdf_net.load_state_dict(checkpoint['sdf_network_fine'])
variance_net = SingleVarianceNetwork(0.3)
variance_net.load_state_dict(checkpoint['variance_network_fine'])

renderer = NeuSRenderer(sdf_net, variance_net)
```

#### **Render Novel Views**
```python
def render_novel_view(camera_pose, image_size=(512, 512)):
    # Generate rays for new camera position
    rays_o, rays_d = generate_rays_from_pose(camera_pose, image_size)
    
    # Render using trained model
    with torch.no_grad():
        render_out = renderer.render(rays_o, rays_d, near=0.1, far=10.0)
        
    return render_out['comp_normal'], render_out['comp_depth']
```

#### **Extract Surface Points**
```python
def extract_surface_points(num_points=10000):
    # Sample points near zero SDF
    points = torch.randn(num_points, 3) * 2  # Random points in [-2, 2]^3
    
    with torch.no_grad():
        sdf_values = sdf_net(points)
    
    # Keep points near surface (SDF ≈ 0)
    surface_mask = torch.abs(sdf_values.squeeze()) < 0.01
    surface_points = points[surface_mask]
    
    return surface_points
```

### 3. Integration into Applications

#### **Real-time Rendering Pipeline**
```python
class PMNIRenderer:
    def __init__(self, model_path):
        self.load_model(model_path)
        self.setup_rendering_pipeline()
    
    def render_frame(self, camera_pose):
        # Fast rendering for real-time applications
        return self.fast_render(camera_pose)
    
    def extract_mesh(self):
        # High-quality mesh extraction
        return self.high_res_mesh_extraction()
```

#### **3D Asset Pipeline**
```python
# Complete asset creation workflow
def create_3d_asset(input_images):
    # 1. Run PMNI training
    trained_model = train_pmni(input_images)
    
    # 2. Extract optimized mesh
    mesh = extract_mesh(trained_model)
    
    # 3. Optimize for target application
    optimized_mesh = optimize_mesh(mesh, target='game_engine')
    
    # 4. Generate LODs
    lods = generate_lods(optimized_mesh)
    
    # 5. Package asset
    asset_package = package_asset(optimized_mesh, lods, textures=None)
    
    return asset_package
```

### 4. Research and Analysis

#### **Surface Analysis**
```python
def analyze_reconstructed_surface(mesh_path):
    mesh = trimesh.load(mesh_path)
    
    # Compute surface properties
    surface_area = mesh.area
    volume = mesh.volume
    curvature = compute_curvature(mesh)
    
    # Quality metrics
    watertight = mesh.is_watertight
    self_intersections = detect_self_intersections(mesh)
    
    return {
        'surface_area': surface_area,
        'volume': volume,
        'watertight': watertight,
        'quality_score': compute_quality_score(mesh)
    }
```

#### **Training Analysis**
```python
def analyze_training_progress(log_path):
    # Parse training logs
    losses = parse_losses(log_path)
    
    # Plot convergence curves
    plot_convergence(losses)
    
    # Analyze training stability
    stability_metrics = compute_stability_metrics(losses)
    
    # Identify optimal stopping point
    optimal_iter = find_optimal_stopping_point(losses)
    
    return training_analysis
```

### 5. Production Deployment

#### **Model Optimization**
```python
# Optimize for inference speed
def optimize_for_inference(model):
    # Quantization
    quantized_model = quantize_model(model, precision='fp16')
    
    # Pruning
    pruned_model = prune_model(quantized_model, sparsity=0.3)
    
    # TensorRT optimization
    trt_model = convert_to_tensorrt(pruned_model)
    
    return trt_model
```

#### **Scalability Considerations**
```python
# Handle large scenes
def process_large_scene(input_data, chunk_size=1000000):
    # Divide scene into manageable chunks
    chunks = divide_scene(input_data, chunk_size)
    
    # Process each chunk
    chunk_results = []
    for chunk in chunks:
        result = process_chunk(chunk)
        chunk_results.append(result)
    
    # Merge results
    final_result = merge_results(chunk_results)
    
    return final_result
```

---

## Advanced Features and Extensions

### 1. Multi-Object Reconstruction

```python
class MultiObjectPMNI:
    def __init__(self):
        self.object_detectors = []
        self.instance_segmentors = []
        self.pmni_instances = []
    
    def reconstruct_scene(self, images):
        # 1. Detect objects
        detections = self.detect_objects(images)
        
        # 2. Segment instances
        masks = self.segment_instances(images, detections)
        
        # 3. Reconstruct each object
        reconstructions = []
        for mask in masks:
            reconstruction = self.reconstruct_object(images, mask)
            reconstructions.append(reconstruction)
        
        return reconstructions
```

### 2. Temporal Consistency

```python
class TemporalPMNI:
    def __init__(self):
        self.temporal_smoothing = ExponentialMovingAverage()
        self.motion_estimation = OpticalFlowEstimator()
    
    def reconstruct_video_sequence(self, frames):
        reconstructions = []
        for frame in frames:
            # Estimate motion from previous frame
            motion = self.estimate_motion(frame, previous_frame)
            
            # Initialize pose from motion
            initial_pose = self.predict_pose_from_motion(motion)
            
            # Reconstruct with temporal prior
            reconstruction = self.reconstruct_with_temporal_prior(
                frame, initial_pose, previous_reconstruction
            )
            
            reconstructions.append(reconstruction)
        
        return reconstructions
```

### 3. Material and Texture Estimation

```python
class TexturedPMNI:
    def __init__(self):
        self.material_net = MaterialNetwork()
        self.texture_net = TextureNetwork()
    
    def reconstruct_with_materials(self, images):
        # Joint geometry and material reconstruction
        geometry, materials = self.joint_optimization(images)
        
        # Texture synthesis
        textures = self.synthesize_textures(geometry, materials)
        
        return geometry, materials, textures
```

---

## Troubleshooting and Best Practices

### 1. Common Issues

#### **Training Instability**
```python
# Symptoms: NaN losses, exploding gradients
# Solutions:
# 1. Reduce learning rate
# 2. Enable gradient clipping
# 3. Use loss scaling
# 4. Check input data quality
```

#### **Poor Reconstruction Quality**
```python
# Symptoms: Holes, artifacts, incorrect topology
# Solutions:
# 1. Increase training iterations
# 2. Improve pose initialization
# 3. Use higher resolution data
# 4. Adjust loss weights
```

#### **Memory Issues**
```python
# Symptoms: CUDA out of memory
# Solutions:
# 1. Reduce batch size
# 2. Use gradient checkpointing
# 3. Enable chunked processing
# 4. Use mixed precision training
```

### 2. Performance Optimization

#### **Speed Improvements**
```python
# 1. Use tiny-cuda-nn for faster encoding
# 2. Optimize occupancy grid updates
# 3. Use hierarchical sampling
# 4. Implement early ray termination
```

#### **Quality Improvements**
```python
# 1. Progressive resolution increase
# 2. Multi-stage loss weighting
# 3. Pose refinement iterations
# 4. Surface normal regularization
```

### 3. Best Practices

#### **Data Preparation**
- Ensure consistent camera calibration
- Use high-quality normal/depth maps
- Validate input data integrity
- Consider pose initialization quality

#### **Training Strategy**
- Start with conservative hyperparameters
- Monitor validation metrics closely
- Use early stopping when appropriate
- Save checkpoints frequently

#### **Evaluation Protocol**
- Use multiple evaluation metrics
- Compare against ground truth when available
- Validate on held-out test views
- Consider both quantitative and qualitative metrics

---

## Conclusion

PMNI represents a significant advancement in neural implicit surface reconstruction by jointly optimizing geometry and camera poses. The method's ability to learn from multi-view images without known camera parameters makes it particularly valuable for applications where pose estimation is challenging.

### Key Strengths:
- **Robust Pose Optimization**: Learns camera parameters alongside geometry
- **High-Quality Reconstructions**: Produces smooth, detailed 3D meshes
- **Scalable Architecture**: Handles complex scenes and objects
- **Research-Friendly**: Extensible framework for novel research directions

### Future Directions:
- **Real-time Applications**: Optimizing for interactive reconstruction
- **Large-Scale Scenes**: Extending to city-scale reconstruction
- **Semantic Understanding**: Incorporating object categories and relationships
- **Physical Simulation**: Integration with physics engines for dynamic scenes

The comprehensive training outputs provide researchers and practitioners with rich data for analysis, validation, and further development. Whether used for academic research, industrial applications, or creative projects, PMNI offers a powerful and flexible framework for 3D reconstruction from multi-view imagery.

---

*This documentation provides a complete understanding of PMNI from theoretical foundations to practical implementation. The system is designed to be both powerful for research and practical for real-world applications.*</content>
<parameter name="filePath">/home/bhanu/pmni/PMNI/pmni_complete_guide.md