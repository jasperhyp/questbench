#!/bin/bash

cd "/n/home09/yeh803/workspace/Reasoning/external/questbench"

# Base path for the data
BASE_SL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"

# 1. IMPORTANT: Restrict each Python job to 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ==========================================
# OUTER LOOP: Iterate sequentially through folders new_0 to new_10
# ==========================================
for i in {1..10}; do

    # Construct the specific directory for this batch (new_0, new_1, etc.)
    SL_DIR="${BASE_SL_DIR}/new_${i}_500k"
    
    # Create a specific log sub-folder so logs don't overwrite each other
    CURRENT_LOG_DIR="logs_generation/new_${i}_500k"
    mkdir -p "$CURRENT_LOG_DIR"

    echo "=================================================="
    echo "Processing Folder: $SL_DIR"
    echo "Logs will be saved to: $CURRENT_LOG_DIR"
    echo "=================================================="

    # ==========================================
    # INNER LOOP: Parallel jobs for indices 0..6900
    # ==========================================
    for start_idx in {0..6900..100}; do

        # Calculate the end index (start + 100)
        end_idx=$((start_idx + 100))

        echo "Launching job for new_${i}: $start_idx -> $end_idx"

        # Run background job & redirect output to the specific log folder
        PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
            --sl_dir "$SL_DIR" \
            --start_idx "$start_idx" \
            --end_idx "$end_idx" \
            --max_k 4 \
            > "${CURRENT_LOG_DIR}/run_${start_idx}_${end_idx}.log" 2>&1 &

    done

    # ==========================================
    # WAIT: Finish all indices for 'new_0' before starting 'new_1'
    # ==========================================
    wait
    echo "Finished processing folder: new_${i}"
    echo "--------------------------------------------------"

done

echo "All folders (new_0 through new_10) completed."