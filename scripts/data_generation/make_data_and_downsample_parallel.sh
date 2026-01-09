#!/bin/bash

# ==========================================
# Usage: ./make_data_and_downsample_parallel.sh <start_folder_index> <end_folder_index> <suffix> <max_parallel_jobs> [max_problems_per_ruleset] [k1_sample] [k2_sample]
# Example: ./make_data_and_downsample_parallel.sh 1 10 500k 4 20 1000 1000
# ==========================================

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <start_folder_index> <end_folder_index> <suffix> <max_parallel_jobs> [max_problems_per_ruleset] [k1_sample] [k2_sample]"
    echo "  suffix: 500k or 1m"
    echo "Example: $0 1 10 500k 4 20 1000 1000"
    exit 1
fi

START_NUM=$1
END_NUM=$2
SUFFIX=$3
MAX_PARALLEL_JOBS=$4
MAX_PROBLEMS_PER_RULESET=${5:-20}
K1_SAMPLE=${6:-2000}
K2_SAMPLE=${7:-2000}

# Validate suffix
if [ "$SUFFIX" != "500k" ] && [ "$SUFFIX" != "1m" ]; then
    echo "Error: suffix must be '500k' or '1m', got '$SUFFIX'"
    exit 1
fi

WORK_DIR="/n/home09/yeh803/workspace/Reasoning/external/questbench"
SCRATCH_SL_DIR="/n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data"
GENERATED_DATA_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"

cd "$WORK_DIR" || { echo "Directory $WORK_DIR not found"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=========================================="
echo "Parallel Make Data and Downsample"
echo "Folders: new_${START_NUM}_${SUFFIX} to new_${END_NUM}_${SUFFIX}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Max problems per ruleset: ${MAX_PROBLEMS_PER_RULESET}"
echo "K1 sample: ${K1_SAMPLE}, K2 sample: ${K2_SAMPLE}"
echo "=========================================="

# Function to process a single folder
process_folder() {
    local folder_idx=$1
    local FOLDER_NAME="new_${folder_idx}_${SUFFIX}"
    local SL_DIR="${SCRATCH_SL_DIR}/${FOLDER_NAME}"
    local DATA_DIR="${GENERATED_DATA_DIR}/${FOLDER_NAME}"
    local LOG_DIR="logs_make_data/${FOLDER_NAME}"
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    
    # Check if source directory exists
    if [ ! -d "$SL_DIR" ]; then
        echo "[${FOLDER_NAME}] Error: Directory $SL_DIR does not exist" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    echo "[${FOLDER_NAME}] Starting pipeline..." >> "${LOG_DIR}/pipeline.log"
    
    # Step 1: Run make_data_new.py
    echo "[${FOLDER_NAME}] Step 1: Running make_data_new.py..." >> "${LOG_DIR}/pipeline.log"
    PYTHONPATH=. python SimpleLogic/make_data_new.py \
        --sl_dir "$SL_DIR" \
        --max_problems_to_sample_per_ruleset "$MAX_PROBLEMS_PER_RULESET" > "${LOG_DIR}/make_data.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[${FOLDER_NAME}] Error: make_data_new.py failed" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    # Step 2: Run downsample.py
    local INPUT_CSV="${SL_DIR}/simplelogic_heldout_k_sufficient_data_new.csv"
    local OUTPUT_CSV="${DATA_DIR}/simplelogic_heldout_k_sufficient_data_new_sampled.csv"
    
    if [ ! -f "$INPUT_CSV" ]; then
        echo "[${FOLDER_NAME}] Error: Input file $INPUT_CSV not found" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    echo "[${FOLDER_NAME}] Step 2: Running downsample.py..." >> "${LOG_DIR}/pipeline.log"
    PYTHONPATH=. python SimpleLogic/downsample.py \
        --input_path "$INPUT_CSV" \
        --output_path "$OUTPUT_CSV" \
        --k1_sample "$K1_SAMPLE" \
        --k2_sample "$K2_SAMPLE" \
        --seed 42 \
        > "${LOG_DIR}/downsample.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[${FOLDER_NAME}] Error: downsample.py failed" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    # Step 3: Verify results
    if [ ! -f "$OUTPUT_CSV" ]; then
        echo "[${FOLDER_NAME}] Error: Output file $OUTPUT_CSV not found" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    echo "[${FOLDER_NAME}] Step 3: Running verify_results.py..." >> "${LOG_DIR}/pipeline.log"
    PYTHONPATH=. python SimpleLogic/tests/verify_results.py \
        --input_csv "$OUTPUT_CSV" \
        > "${LOG_DIR}/verify.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[${FOLDER_NAME}] Error: verify_results.py failed" >> "${LOG_DIR}/pipeline.log"
        return 1
    fi
    
    echo "[${FOLDER_NAME}] Pipeline completed successfully!" >> "${LOG_DIR}/pipeline.log"
    return 0
}

# Export function and variables for subshells
export -f process_folder
export SUFFIX SCRATCH_SL_DIR GENERATED_DATA_DIR MAX_PROBLEMS_PER_RULESET K1_SAMPLE K2_SAMPLE

# Counter for parallel batching
job_counter=0

for ((i=START_NUM; i<=END_NUM; i++)); do
    FOLDER_NAME="new_${i}_${SUFFIX}"
    echo "Launching: ${FOLDER_NAME}"
    
    # Launch background job
    process_folder "$i" &
    
    ((job_counter++))
    
    # Throttling logic
    if (( job_counter % MAX_PARALLEL_JOBS == 0 )); then
        echo "[Batch Limit Reached] Waiting for current batch of $MAX_PARALLEL_JOBS jobs to finish..."
        wait
        echo "[Resuming] Launching next batch..."
    fi
done

# Wait for remaining jobs
wait

echo "=========================================="
echo "All folders processed!"
echo "Check logs_make_data/new_*_${SUFFIX}/pipeline.log for status of each folder"
echo "=========================================="
