#!/bin/bash
cd /home/bhanu/pmni/PMNI
source /home/bhanu/miniconda3/etc/profile.d/conda.sh
conda activate PMNI
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nohup python exp_runner.py --conf config/diligent_bear.conf --obj_name bear > logs/train_${TIMESTAMP}.log 2>&1 &
PID=$!
echo $PID > logs/train_${TIMESTAMP}.pid
echo "Started training with PID $PID"
echo "Log: logs/train_${TIMESTAMP}.log"
echo "PID file: logs/train_${TIMESTAMP}.pid"
