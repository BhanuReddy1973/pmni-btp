#!/bin/bash

# PMNI Training Monitor
# Real-time monitoring of training progress with health checks

LOG_DIR="logs"
REFRESH_INTERVAL=5  # seconds

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Find latest log file
find_latest_log() {
    ls -t ${LOG_DIR}/train_*.log 2>/dev/null | head -1
}

# Extract metric from log line
extract_metric() {
    local line=$1
    local metric=$2
    echo "$line" | grep -oP "${metric}=\K[0-9.e+-]+" | head -1
}

# Format number with color based on health
format_metric() {
    local value=$1
    local metric_name=$2
    local threshold_warning=$3
    local threshold_critical=$4
    
    if [ -z "$value" ]; then
        echo "${RED}N/A${NC}"
        return
    fi
    
    # Compare with thresholds (if provided)
    if [ -n "$threshold_critical" ] && [ $(echo "$value > $threshold_critical" | bc -l) -eq 1 ]; then
        echo "${RED}${value}${NC}"
    elif [ -n "$threshold_warning" ] && [ $(echo "$value > $threshold_warning" | bc -l) -eq 1 ]; then
        echo "${YELLOW}${value}${NC}"
    else
        echo "${GREEN}${value}${NC}"
    fi
}

# Main monitoring loop
monitor_training() {
    local log_file=$(find_latest_log)
    
    if [ -z "$log_file" ]; then
        echo -e "${RED}No training log found in ${LOG_DIR}/${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         PMNI Training Monitor - Press Ctrl+C to exit          ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Monitoring: $log_file${NC}"
    echo ""
    
    local prev_iter=0
    local start_time=$(date +%s)
    
    while true; do
        clear
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║              PMNI Training Monitor (Live Update)              ║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Get last training line with metrics
        local last_line=$(grep -E "loss=" "$log_file" | tail -1)
        
        if [ -z "$last_line" ]; then
            echo -e "${YELLOW}Waiting for training to start...${NC}"
            sleep $REFRESH_INTERVAL
            continue
        fi
        
        # Extract iteration and progress
        local iter=$(echo "$last_line" | grep -oP '\s+\K[0-9]+(?=/[0-9]+)' | tail -1)
        local total_iter=$(echo "$last_line" | grep -oP '/\K[0-9]+' | tail -1)
        local progress_pct=$(echo "scale=1; $iter * 100 / $total_iter" | bc)
        
        # Calculate speed
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local iter_per_sec=$(echo "scale=2; ($iter - $prev_iter) / $REFRESH_INTERVAL" | bc)
        
        # Estimate time remaining
        local iters_left=$((total_iter - iter))
        local eta_seconds=$(echo "scale=0; $iters_left / $iter_per_sec" | bc 2>/dev/null || echo "0")
        local eta_hours=$((eta_seconds / 3600))
        local eta_mins=$(((eta_seconds % 3600) / 60))
        
        # Progress bar
        local bar_width=50
        local filled=$(echo "scale=0; $bar_width * $iter / $total_iter" | bc)
        local empty=$((bar_width - filled))
        local bar=$(printf "%${filled}s" | tr ' ' '█')
        local empty_bar=$(printf "%${empty}s" | tr ' ' '░')
        
        echo -e "${MAGENTA}Progress: [${bar}${empty_bar}] ${progress_pct}%${NC}"
        echo -e "${BLUE}Iteration: ${iter} / ${total_iter}${NC}"
        echo -e "${BLUE}Speed: ${iter_per_sec} it/s${NC}"
        echo -e "${BLUE}ETA: ${eta_hours}h ${eta_mins}m${NC}"
        echo ""
        
        # Extract metrics
        local loss=$(extract_metric "$last_line" "loss")
        local normal=$(extract_metric "$last_line" "normal")
        local depth=$(extract_metric "$last_line" "depth")
        local mask=$(extract_metric "$last_line" "mask")
        local eikonal=$(extract_metric "$last_line" "eikonal")
        local s_val=$(extract_metric "$last_line" "s")
        local samples=$(extract_metric "$last_line" "samples_per_ray")
        
        echo -e "${CYAN}┌─ Metrics ──────────────────────────────────────────────────┐${NC}"
        printf "│ %-20s: %30s │\n" "Total Loss" "$(format_metric "$loss" "loss" 10 100)"
        printf "│ %-20s: %30s │\n" "Normal Loss" "$(format_metric "$normal" "normal")"
        printf "│ %-20s: %30s │\n" "Depth Loss" "$(format_metric "$depth" "depth")"
        printf "│ %-20s: %30s │\n" "Mask Loss" "$(format_metric "$mask" "mask")"
        printf "│ %-20s: %30s │\n" "Eikonal Loss" "$(format_metric "$eikonal" "eikonal")"
        printf "│ %-20s: %30s │\n" "Deviation Scale" "$(format_metric "$s_val" "s")"
        printf "│ %-20s: %30s │\n" "Samples/Ray" "$(format_metric "$samples" "samples")"
        echo -e "${CYAN}└────────────────────────────────────────────────────────────┘${NC}"
        echo ""
        
        # Check for warnings
        local warnings=0
        
        if [ -n "$samples" ] && [ $(echo "$samples < 0.5" | bc -l) -eq 1 ]; then
            echo -e "${RED}⚠ WARNING: Samples per ray is critically low!${NC}"
            warnings=$((warnings + 1))
        fi
        
        if [ -n "$loss" ] && [ $(echo "$loss > 100" | bc -l) -eq 1 ]; then
            echo -e "${RED}⚠ WARNING: Loss is extremely high!${NC}"
            warnings=$((warnings + 1))
        fi
        
        # Check for recent errors
        local recent_errors=$(tail -100 "$log_file" | grep -i "error\|nan\|traceback" | wc -l)
        if [ $recent_errors -gt 0 ]; then
            echo -e "${RED}⚠ WARNING: $recent_errors error(s) detected in recent log!${NC}"
            warnings=$((warnings + 1))
        fi
        
        if [ $warnings -eq 0 ]; then
            echo -e "${GREEN}✓ Training appears healthy${NC}"
        fi
        
        echo ""
        echo -e "${CYAN}Recent Log Output:${NC}"
        echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
        tail -5 "$log_file" | head -4
        echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
        
        echo ""
        echo -e "${BLUE}Last update: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo -e "${BLUE}Refreshing in ${REFRESH_INTERVAL}s... (Press Ctrl+C to exit)${NC}"
        
        prev_iter=$iter
        sleep $REFRESH_INTERVAL
    done
}

# Start monitoring
monitor_training
