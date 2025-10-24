# PMNI Setup Guide

Complete setup and usage instructions for the **Pose-free Multi-view Normal Integration (PMNI)** project.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Setup](#dataset-setup)
4. [Verification](#verification)
5. [Running Training](#running-training)
6. [Troubleshooting](#troubleshooting)
7. [Project Structure](#project-structure)

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (tested on H100 with 19.6GB VRAM)
- **CUDA**: Version 12.1 (system CUDA)
- **Storage**: ~2GB for dataset + ~5GB for environment

### Required Software
- NVIDIA Driver 555.52.04 or compatible
- System CUDA toolkit 12.1
- Git for cloning repositories
- Internet connection for downloading datasets and packages

---

## 🚀 Environment Setup

### Step 1: Install Miniconda

If you don't have Miniconda installed:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
```

### Step 2: Run Setup Script

The project includes a setup script that installs all dependencies:

```bash
cd /home/bhanu/pmni/PMNI
bash setup_clean.sh
```

**What the script does:**
- Creates a conda environment named `PMNI` with Python 3.8
- Installs PyTorch 2.1.0 with CUDA 12.1 support
- Compiles and installs tiny-cuda-nn (CUDA extension)
- Installs pytorch3d 0.7.8 (with memory-safe compilation)
- Installs all required Python packages from `requirements.txt`
- Installs nerfacc from local third_parties directory

**Expected time**: 15-30 minutes depending on your system

### Step 3: Activate Environment

Always activate the PMNI environment before working:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI
```

To make this easier, add an alias to your `~/.bashrc`:

```bash
echo "alias pmni='source ~/miniconda3/etc/profile.d/conda.sh && conda activate PMNI'" >> ~/.bashrc
source ~/.bashrc
```

Now you can simply run: `pmni`

---

## 📦 Dataset Setup

### Automatic Download

The dataset is automatically downloaded during setup. If you need to download it manually:

```bash
cd /home/bhanu/pmni/PMNI
pip install gdown
gdown 1nfWRA3OvPAlNNrj6kUfFDgDX_FXlf5S7
unzip data.zip
rm data.zip
```

### Dataset Structure

After extraction, your dataset should be organized as:

```
data/data/
├── diligent_mv_normals/
│   ├── bear/
│   ├── buddha/
│   ├── cow/
│   ├── pot2/
│   └── reading/
├── own_objects_normals/
│   ├── cat/
│   ├── dog/
│   ├── dragon/
│   ├── monkey/
│   ├── pineapple/
│   └── tiger/
└── special_normals/
    ├── ball/
    └── cylinder/
```

Each object folder contains:
- `K.txt` - Camera intrinsics
- `cameras_sphere.npz` - Camera poses
- `*.exr` - Multi-view normal maps
- `mesh_Gt.ply` - Ground truth mesh

---

## ✅ Verification

### Verify CUDA and PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA H100 PCIe
```

### Verify All Libraries

Run this comprehensive check:

```bash
python << 'EOF'
import sys
libraries = [
    ('torch', 'torch.__version__'),
    ('torchvision', 'torchvision.__version__'),
    ('torchaudio', 'torchaudio.__version__'),
    ('tinycudann', 'tinycudann.__version__'),
    ('pytorch3d', 'pytorch3d.__version__'),
    ('trimesh', 'trimesh.__version__'),
    ('open3d', 'o3d.__version__'),
    ('nerfacc', 'nerfacc.__version__'),
    ('cv2', 'cv2.__version__'),
    ('PIL', 'PIL.__version__'),
    ('pyexr', 'pyexr.__version__'),
    ('imageio', 'imageio.__version__'),
    ('skimage', 'skimage.__version__'),
    ('numpy', 'np.__version__'),
    ('scipy', 'scipy.__version__'),
    ('pyvista', 'pyvista.__version__'),
    ('matplotlib', 'matplotlib.__version__'),
    ('tqdm', 'tqdm.__version__'),
    ('tensorboard', None),
    ('yaml', 'yaml.__version__'),
    ('pyhocon', None),
    ('icecream', None),
    ('mcubes', None),
    ('pyembree', None),
]

passed = 0
failed = 0

for module_name, version_attr in libraries:
    try:
        if module_name == 'cv2':
            import cv2
            print(f'✅ opencv: {cv2.__version__}')
        elif module_name == 'PIL':
            from PIL import Image
            import PIL
            print(f'✅ Pillow: {PIL.__version__}')
        elif module_name == 'skimage':
            import skimage
            print(f'✅ scikit-image: {skimage.__version__}')
        elif module_name == 'pyexr':
            import pyexr
            print(f'✅ pyexr: {pyexr.__version__}')
        elif module_name == 'open3d':
            import open3d as o3d
            print(f'✅ open3d: {o3d.__version__}')
        else:
            exec(f'import {module_name}')
            if version_attr:
                version = eval(version_attr)
                print(f'✅ {module_name}: {version}')
            else:
                print(f'✅ {module_name}')
        passed += 1
    except Exception as e:
        print(f'❌ {module_name}: {str(e)}')
        failed += 1

print(f'\n{"="*50}')
print(f'RESULTS: {passed}/{len(libraries)} PASSED')
if failed == 0:
    print('✨ Environment is READY! ✨')
else:
    print(f'⚠️  {failed} libraries failed. Please reinstall.')
EOF
```

---

## 🎯 Running Training

### Basic Training Command

Train on a specific object from the DiLiGenT dataset:

```bash
# Activate environment first
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI

# Run training on bear object
CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear
```

### Configuration Files

The project includes several pre-configured training setups:

| Config File | Description | Objects |
|------------|-------------|---------|
| `config/diligent_bear.conf` | Single object (bear) | bear |
| `config/diligent.conf` | Full DiLiGenT dataset | bear, buddha, cow, pot2, reading |
| `config/own_objects.conf` | Custom objects | cat, dog, dragon, monkey, pineapple, tiger |
| `config/ball.conf` | Synthetic sphere | ball |
| `config/cylinder.conf` | Synthetic cylinder | cylinder |

### Training Examples

**Example 1: Train on single object (bear)**
```bash
CUDA_VISIBLE_DEVICES=0 python exp_runner.py \
    --conf config/diligent_bear.conf \
    --obj_name bear
```

**Example 2: Train on all DiLiGenT objects**
```bash
CUDA_VISIBLE_DEVICES=0 python exp_runner.py \
    --conf config/diligent.conf
```

**Example 3: Train on custom object (cat)**
```bash
CUDA_VISIBLE_DEVICES=0 python exp_runner.py \
    --conf config/own_objects.conf \
    --obj_name cat
```

### Multi-GPU Training

If you have multiple GPUs, specify which one to use:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear

# Use multiple GPUs (0 and 1)
CUDA_VISIBLE_DEVICES=0,1 python exp_runner.py --conf config/diligent.conf
```

### Output Structure

Training outputs are saved in the experiment directory:

```
exps/<experiment_name>/<object_name>/
├── checkpoints/          # Model checkpoints
├── plots/                # Visualization plots
├── logs/                 # Training logs
└── reconstructions/      # Final 3D meshes
```

---

## 🔍 Troubleshooting

### Issue: Import errors after setup

**Solution**: Make sure you've activated the conda environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or resolution in config files:
```
training:
    batch_size = 512  # Try reducing to 256 or 128
```

### Issue: tiny-cuda-nn compilation fails

**Solution**: Verify CUDA versions match:
```bash
# Check system CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

They should both be 12.1. If not, reinstall PyTorch:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: open3d import error (libGL.so.1)

**Solution**: Install system OpenGL libraries:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Issue: Container crashes during pytorch3d installation

**Solution**: The setup script already handles this with `MAX_JOBS=2`. If you need to reinstall manually:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation --verbose
```

### Issue: Training exits with code 1

**Possible causes**:
1. Check dataset path in config file
2. Verify object name matches folder name
3. Check GPU memory availability
4. Review error logs in terminal output

**Debug mode**: Add `--debug` flag for verbose output:
```bash
python exp_runner.py --conf config/diligent_bear.conf --obj_name bear --debug
```

---

## 📁 Project Structure

```
PMNI/
├── config/                    # Configuration files
│   ├── diligent_bear.conf    # Single object config
│   ├── diligent.conf         # DiLiGenT dataset config
│   ├── own_objects.conf      # Custom objects config
│   ├── ball.conf             # Synthetic ball
│   └── cylinder.conf         # Synthetic cylinder
│
├── data/                      # Dataset directory
│   └── data/                 # Actual dataset
│       ├── diligent_mv_normals/
│       ├── own_objects_normals/
│       └── special_normals/
│
├── models/                    # Core model implementations
│   ├── __init__.py
│   ├── common.py             # Common utilities
│   ├── dataset_loader.py     # Dataset loading
│   ├── fields.py             # Neural fields (SDF, color)
│   ├── init_pose.py          # Pose initialization
│   ├── losses.py             # Loss functions
│   ├── pose_net.py           # Pose network
│   ├── posenet2.py           # Alternative pose network
│   ├── rend_util.py          # Rendering utilities
│   ├── renderer.py           # Volume renderer
│   ├── scale_net.py          # Scale network
│   └── visibility_tracer.py  # Visibility tracing
│
├── third_parties/             # External dependencies
│   └── nerfacc-0.3.5/        # NeRF acceleration library
│
├── utilities/                 # Utility functions
│   └── utils.py              # General utilities
│
├── fig/                       # Figures and assets
│
├── exp_runner.py             # Main training script
├── requirements.txt          # Python dependencies
├── setup_clean.sh            # Environment setup script
├── README.md                 # Project README
└── SETUP_GUIDE.md           # This file
```

---

## 🎓 Additional Resources

### Paper & Citation

If you use this code, please cite:

```bibtex
@inproceedings{lyu2023pmni,
  title={Pose-free Multi-view Normal Integration},
  author={Lyu, Xu and Zhang, Jingyang and Wang, Peng and others},
  booktitle={CVPR},
  year={2023}
}
```

### Key Dependencies

- **PyTorch 2.1.0**: Deep learning framework
- **tiny-cuda-nn**: Fast CUDA neural networks
- **pytorch3d**: 3D deep learning operations
- **nerfacc**: Neural radiance field acceleration
- **Open3D**: 3D data processing
- **Trimesh**: 3D mesh processing

### Important Notes

1. **CUDA Version Matching**: This setup uses PyTorch with CUDA 12.1 to match system CUDA. Do not mix CUDA versions.

2. **Memory Requirements**: 
   - Minimum: 8GB GPU VRAM
   - Recommended: 16GB+ GPU VRAM for full resolution

3. **Training Time**: 
   - Single object: ~2-4 hours on H100
   - Full dataset: ~10-20 hours depending on GPU

4. **Data Format**: 
   - Normal maps: `.exr` format (HDR)
   - Meshes: `.ply` format
   - Camera parameters: `.npz` and `.txt` files

---

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check the original repository issues: [GitHub Issues](https://github.com/pmz-enterprise/PMNI/issues)
2. Verify your environment matches the requirements
3. Review the troubleshooting section above
4. Check CUDA and PyTorch compatibility

---

## ✅ Quick Start Checklist

- [ ] Miniconda installed at `~/miniconda3`
- [ ] PMNI conda environment created and activated
- [ ] All 24 libraries verified working
- [ ] Dataset downloaded and extracted to `data/data/`
- [ ] GPU detected and CUDA working
- [ ] Successfully run training on test object

**Ready to train?** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI
CUDA_VISIBLE_DEVICES=0 python exp_runner.py --conf config/diligent_bear.conf --obj_name bear
```

---

**Last Updated**: October 24, 2025  
**Environment**: PMNI Conda Environment with Python 3.8, PyTorch 2.1.0+cu121, CUDA 12.1
