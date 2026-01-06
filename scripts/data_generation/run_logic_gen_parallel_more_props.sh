#!/bin/bash

# ==========================================
# INPUT VALIDATION
# ==========================================
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_folder_index> <end_folder_index> <max_expansions_per_layer>"
    echo "Example: $0 1 10 500000"
    exit 1
fi

START_NUM=$1
END_NUM=$2
MAX_EXPANSIONS_PER_LAYER=$3

cd "/n/home09/yeh803/workspace/Reasoning/external/questbench"

# Base path for the data
BASE_SL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"

# 1. IMPORTANT: Restrict each Python job to 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running batch process from new_${START_NUM} to new_${END_NUM}"

# ==========================================
# OUTER LOOP: Iterate from START_NUM to END_NUM
# ==========================================
for ((i=START_NUM; i<=END_NUM; i++)); do

    # Construct the specific directory for this batch (new_0, new_1, etc.)
    if [ "$MAX_EXPANSIONS_PER_LAYER" -eq 500000 ]; then
        SL_DIR="${BASE_SL_DIR}/new_${i}_500k"
    elif [ "$MAX_EXPANSIONS_PER_LAYER" -eq 1000000 ]; then
        SL_DIR="${BASE_SL_DIR}/new_${i}_1m"
    else
        echo "Error: Unsupported max_expansions_per_layer value: $MAX_EXPANSIONS_PER_LAYER"
        exit 1
    fi
    
    # Create a specific log sub-folder so logs don't overwrite each other
    # CURRENT_LOG_DIR="logs_generation/new_${i}_500k"
    if [ "$MAX_EXPANSIONS_PER_LAYER" -eq 500000 ]; then
        CURRENT_LOG_DIR="logs_generation/new_${i}_500k"
    elif [ "$MAX_EXPANSIONS_PER_LAYER" -eq 1000000 ]; then
        CURRENT_LOG_DIR="logs_generation/new_${i}_1m"
    fi
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
            --max_expansions_per_layer $MAX_EXPANSIONS_PER_LAYER \
            > "${CURRENT_LOG_DIR}/run_${start_idx}_${end_idx}.log" 2>&1 &

    done

    # ==========================================
    # WAIT: Finish all indices for current folder before starting next folder
    # ==========================================
    wait
    echo "Finished processing folder: new_${i}"
    echo "--------------------------------------------------"

done

echo "All folders (new_${START_NUM} through new_${END_NUM}) completed."