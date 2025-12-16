#!/bin/bash

# Define the data directory path to keep the command clean
SL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"
LOG_DIR="logs_generation"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# 1. IMPORTANT: Restrict each Python job to 1 CPU core
#    to prevent 35 jobs from fighting over resources and freezing the server.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 2. Loop from 0 to 6800 with increments of 200
for start_idx in {0..6800..200}; do

    # Calculate the end index (start + 200)
    end_idx=$((start_idx + 200))

    echo "Launching job: $start_idx -> $end_idx (Log: $LOG_DIR/run_${start_idx}_${end_idx}.log)"

    # Run background job & redirect output to a specific log file
    PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
        --sl_dir "$SL_DIR" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        --max_k 4 \
        > "${LOG_DIR}/run_${start_idx}_${end_idx}.log" 2>&1 &

done

# Wait for all background jobs to finish before exiting the script
wait
echo "All jobs completed."
