#!/usr/bin/env bash
set -euo pipefail

echo "=== PMNI Environment Setup (CUDA 12.1 Compatible) ==="
echo "This script creates a conda environment matching your system CUDA 12.1"

# Check for conda
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda not found. Please install Miniconda first." >&2
  exit 1
fi

# Check for nvcc
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. CUDA toolkit is required." >&2
  exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "Detected system CUDA version: $CUDA_VERSION"

# Remove existing environment
conda deactivate || true
echo "Removing existing PMNI environment (if any)..."
conda remove -y -n PMNI --all 2>/dev/null || true

# Create environment with Python 3.8
echo "Creating conda environment 'PMNI' with Python 3.8..."
conda create -y -n PMNI python=3.8

# Install PyTorch matching system CUDA (12.1)
echo "Installing PyTorch 2.1.0 with CUDA 12.1..."
conda run -n PMNI pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install tiny-cuda-nn (will use system CUDA 12.1)
echo "Installing tiny-cuda-nn with system CUDA 12.1..."
conda run -n PMNI pip install git+https://github.com/NVlabs/tiny-cuda-nn@2ec562e853e6f482b5d09168705205f46358fb39#subdirectory=bindings/torch

# Install nerfacc
echo "Installing nerfacc..."
conda run -n PMNI pip install -e ./third_parties/nerfacc-0.3.5/nerfacc-0.3.5/

# Install other dependencies
echo "Installing other Python dependencies..."
conda run -n PMNI pip install \
    opencv-python==4.8.1.78 \
    trimesh==3.23.5 \
    open3d==0.17 \
    pyvista==0.42.3 \
    scipy==1.10.1 \
    scikit-image==0.21.0 \
    pyhocon==0.3.59 \
    pyexr==0.3.10 \
    tensorboard==2.14.0 \
    icecream==2.1.3 \
    PyMCubes==0.1.4 \
    pyembree==0.2.11

# Install pytorch3d
echo "Installing pytorch3d..."
conda run -n PMNI pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo ""
echo "=== Setup Complete! ==="
echo "Activate with: conda activate PMNI"
echo ""
echo "Verify installation:"
echo "  conda activate PMNI"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo "  python -c 'import tinycudann; print(\"tiny-cuda-nn: OK\")'"
