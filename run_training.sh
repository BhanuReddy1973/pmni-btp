#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PMNI

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory
mkdir -p logs

# Run training with all arguments passed to this script
CUDA_VISIBLE_DEVICES=0 python exp_runner.py "$@" > "logs/train_${TIMESTAMP}.log" 2>&1

# Print status
echo "Training started at ${TIMESTAMP}"
echo "Log file: logs/train_${TIMESTAMP}.log"
echo "Process ID: $$"
