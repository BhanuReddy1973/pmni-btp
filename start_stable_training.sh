
#!/bin/bash

# PMNI Stable Training Launcher
# This script ensures all prerequisites are met before starting training
# and automatically handles recovery from crashes

set -e  # Exit on error during setup

echo "================================================"
echo "PMNI Stable Training Launcher"
echo "================================================"
echo ""

# Activate PMNI conda environment
echo "Activating PMNI environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI
echo "✓ Environment activated"

# Configuration
CONF_FILE="config/diligent_bear.conf"
OBJ_NAME="bear"
LOG_DIR="logs"
CHECKPOINT_DIR="exp/diligent_mv/${OBJ_NAME}"

# Create log directory
mkdir -p ${LOG_DIR}

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

echo "Step 1: Checking Prerequisites..."
echo "-----------------------------------"

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi
echo "✓ Python: $(python --version)"

# Check PyTorch and CUDA
CUDA_CHECK=$(python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1)
if [ "$CUDA_CHECK" != "CUDA" ]; then
    echo "WARNING: CUDA not available! Training will fail."
    echo "Please ensure you are running on a machine with GPU access."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ CUDA available"
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi

# Check for existing checkpoints
echo ""
echo "Step 2: Checking for existing checkpoints..."
echo "-----------------------------------"
LATEST_CKPT=$(find ${CHECKPOINT_DIR} -name "ckpt_*.pth" -type f 2>/dev/null | sort -V | tail -1 || echo "")
if [ -n "$LATEST_CKPT" ]; then
    CKPT_ITER=$(basename "$LATEST_CKPT" | sed 's/ckpt_\([0-9]*\).pth/\1/')
    echo "Found checkpoint: $LATEST_CKPT (iteration $CKPT_ITER)"
    echo "Training will automatically resume from this checkpoint."
else
    echo "No checkpoints found. Starting fresh training."
fi

echo ""
echo "Step 3: Verifying configuration..."
echo "-----------------------------------"
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONF_FILE"
    exit 1
fi
echo "✓ Config file: $CONF_FILE"

# Extract key config values
END_ITER=$(grep -E "^\s*end_iter\s*=" "$CONF_FILE" | sed 's/.*=\s*\([0-9]*\).*/\1/')
SAVE_FREQ=$(grep -E "^\s*save_freq\s*=" "$CONF_FILE" | sed 's/.*=\s*\([0-9]*\).*/\1/')
BATCH_SIZE=$(grep -E "^\s*batch_size\s*=" "$CONF_FILE" | sed 's/.*=\s*\([0-9]*\).*/\1/')

echo "  End iteration: $END_ITER"
echo "  Save frequency: $SAVE_FREQ"
echo "  Batch size: $BATCH_SIZE"

# Calculate expected checkpoints
NUM_CKPTS=$((END_ITER / SAVE_FREQ))
echo "  Expected checkpoints: $NUM_CKPTS"

echo ""
echo "Step 4: Starting training..."
echo "-----------------------------------"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

# Start training in background with proper error handling
(
    # Trap errors
    trap 'echo "ERROR: Training crashed! Check log: $LOG_FILE"; exit 1' ERR
    
    # Run training
    python exp_runner.py \
        --conf "$CONF_FILE" \
        --obj_name "$OBJ_NAME" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "================================================"
        echo "SUCCESS: Training completed!"
        echo "================================================"
        echo "Final checkpoint: ${CHECKPOINT_DIR}/checkpoints/ckpt_${END_ITER}.pth"
        echo "Logs saved to: $LOG_FILE"
    else
        echo ""
        echo "================================================"
        echo "ERROR: Training exited with code $TRAIN_EXIT_CODE"
        echo "================================================"
        echo "Check log file: $LOG_FILE"
        exit $TRAIN_EXIT_CODE
    fi
) &

# Save PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "Training started with PID: $TRAIN_PID"
echo ""
echo "To monitor training in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo "  # or: kill \$(cat $PID_FILE)"
echo ""
echo "To check training status:"
echo "  ps -p $TRAIN_PID"
echo ""
echo "================================================"
