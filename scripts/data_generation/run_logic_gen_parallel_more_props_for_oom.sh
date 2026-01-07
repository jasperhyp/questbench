#!/bin/bash

# ==========================================
# 1. INPUT VALIDATION & SETUP
# ==========================================
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start_folder_index> <end_folder_index> <max_expansions_per_layer> <max_parallel_jobs>"
    exit 1
fi

START_NUM=$1
END_NUM=$2
MAX_EXPANSIONS_PER_LAYER=$3
MAX_PARALLEL_JOBS=$4

WORK_DIR="/n/home09/yeh803/workspace/Reasoning/external/questbench"
BASE_SL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"

cd "$WORK_DIR" || { echo "Directory $WORK_DIR not found"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ==========================================
# 2. BATCH PROCESS
# ==========================================
for ((i=START_NUM; i<=END_NUM; i++)); do

    if [ "$MAX_EXPANSIONS_PER_LAYER" -eq 500000 ]; then
        SUFFIX="500k"
    elif [ "$MAX_EXPANSIONS_PER_LAYER" -eq 1000000 ]; then
        SUFFIX="1m"
    else
        echo "Error: Unsupported max expansions: $MAX_EXPANSIONS_PER_LAYER"
        exit 1
    fi

    SL_DIR="${BASE_SL_DIR}/new_${i}_${SUFFIX}"
    CURRENT_LOG_DIR="logs_generation/new_${i}_${SUFFIX}"
    mkdir -p "$CURRENT_LOG_DIR"

    echo "Processing Folder: new_${i} ($SUFFIX)"
    
    # Counter for the parallel batching
    job_counter=0

    for start_idx in {0..6900..100}; do
        end_idx=$((start_idx + 100))

        # Launch background job
        PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
            --sl_dir "$SL_DIR" \
            --start_idx "$start_idx" \
            --end_idx "$end_idx" \
            --max_k 4 \
            --max_expansions_per_layer "$MAX_EXPANSIONS_PER_LAYER" \
            > "${CURRENT_LOG_DIR}/run_${start_idx}_${end_idx}.log" 2>&1 &
        
        # Increment counter
        ((job_counter++))

        # === THROTTLING LOGIC ===
        # If we have launched MAX_PARALLEL_JOBS, wait for them to finish.
        if (( job_counter % MAX_PARALLEL_JOBS == 0 )); then
            echo "  [Batch Limit Reached] Waiting for current batch of $MAX_PARALLEL_JOBS jobs to finish..."
            wait
            echo "  [Resuming] Launching next batch..."
        fi

    done

    # Final wait for any remaining jobs in the last batch
    wait
    echo "Finished processing folder: new_${i}"
    echo "--------------------------------------------------"

done

echo "All Done."