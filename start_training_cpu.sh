#!/bin/bash

# PMNI CPU Training Launcher (with nohup background execution)
# WARNING: This will be VERY slow on CPU and may fail due to CUDA dependencies

set +e  # Don't exit on error

echo "================================================"
echo "PMNI CPU Training Launcher (Background Mode)"
echo "================================================"
echo ""
echo "WARNING: Training on CPU will be extremely slow!"
echo "Expect 100-1000x slower than GPU training."
echo "Estimated time: 500-5000 hours for 30k iterations"
echo ""

# Configuration
CONF_FILE="config/diligent_bear.conf"
OBJ_NAME="bear"
LOG_DIR="logs"

# Create log directory
mkdir -p ${LOG_DIR}

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

echo "Step 1: Checking environment..."
echo "-----------------------------------"

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi
echo "✓ Python: $(python --version)"

# Check PyTorch
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ PyTorch: $TORCH_VERSION"
else
    echo "✗ PyTorch not found!"
    exit 1
fi

# Warn about CUDA
CUDA_CHECK=$(python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1)
echo "  Device: $CUDA_CHECK"
if [ "$CUDA_CHECK" = "CPU" ]; then
    echo ""
    echo "⚠️  WARNING: Running on CPU only!"
    echo "   This will be extremely slow and may fail."
    echo ""
fi

echo ""
# Step 2: Environment sanity checks for CPU-only mode
echo "Step 2: Performing CPU-only compatibility checks..."
echo "-----------------------------------"

# Check tinycudann availability (not required if running CPU-only, but the code imports it)
python - << 'PY'
try:
    import torch
    import importlib
    cuda_ok = torch.cuda.is_available()
    tcnn_spec = importlib.util.find_spec('tinycudann')
    pypose_spec = importlib.util.find_spec('pypose')
    nerfacc_spec = importlib.util.find_spec('nerfacc')
    if not cuda_ok:
        print('[FATAL] CUDA not available. PMNI training requires GPU for nerfacc & tinycudann kernels.')
        print('        Please run on a GPU-enabled machine. Exiting without starting training.')
        raise SystemExit(2)
    if tcnn_spec is None:
        print('[FATAL] tinycudann is not installed. Install it in your CUDA-enabled environment.')
        raise SystemExit(1)
    if nerfacc_spec is None:
        print('[FATAL] nerfacc is not installed. Install it in your CUDA-enabled environment.')
        raise SystemExit(1)
    if pypose_spec is None:
        print('[WARN] pypose is not installed. If pose learning is enabled, install pypose to avoid runtime errors.')
except SystemExit as e:
    raise
except Exception as e:
    print('[FATAL] Unexpected environment error:', e)
    raise SystemExit(1)
PY

if [ $? -ne 0 ]; then
    echo "Exiting due to environment incompatibility before launch."
    exit 1
fi

echo "Environment OK (CUDA + dependencies present). Launching training..."
echo "Step 3: Starting training in background..."
echo "-----------------------------------"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

nohup python -u exp_runner.py \
        --conf "$CONF_FILE" \
        --obj_name "$OBJ_NAME" \
        > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "✓ Training started in background with PID: $TRAIN_PID"
echo ""
echo "Monitor: tail -f $LOG_FILE"
echo "Check:   ps -p $TRAIN_PID"
echo "Stop:    kill $TRAIN_PID"
echo "================================================"
