#!/bin/bash

# ==========================================
# Usage: ./make_data_and_downsample.sh <folder_name> [max_problems_per_ruleset] [k1_sample] [k2_sample]
# Example: ./make_data_and_downsample.sh new_11_500k 20 1000 1000
# ==========================================

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <folder_name> [max_problems_per_ruleset] [k1_sample] [k2_sample]"
    echo "Example: $0 new_11_500k 50 1000 1000"
    exit 1
fi

FOLDER_NAME=$1
MAX_PROBLEMS_PER_RULESET=${2:-20}
K1_SAMPLE=${3:-1000}
K2_SAMPLE=${4:-1000}

WORK_DIR="/n/home09/yeh803/workspace/Reasoning/external/questbench"
SCRATCH_SL_DIR="/n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data"
GENERATED_DATA_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"
SL_DIR="${SCRATCH_SL_DIR}/${FOLDER_NAME}"
DATA_DIR="${GENERATED_DATA_DIR}/${FOLDER_NAME}"

# Check if the folder exists
if [ ! -d "$SL_DIR" ]; then
    echo "Error: Directory $SL_DIR does not exist"
    exit 1
fi

cd "$WORK_DIR" || { echo "Directory $WORK_DIR not found"; exit 1; }

LOG_DIR="logs_make_data/${FOLDER_NAME}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Processing folder: ${FOLDER_NAME}"
echo "SL_DIR: ${SL_DIR}"
echo "Max problems per ruleset: ${MAX_PROBLEMS_PER_RULESET}"
echo "K1 sample size: ${K1_SAMPLE}"
echo "K2 sample size: ${K2_SAMPLE}"
echo "=========================================="

# Step 1: Run make_data_new.py
echo "[Step 1] Running make_data_new.py..."
PYTHONPATH=. python SimpleLogic/make_data_new.py \
    --sl_dir "$SL_DIR" \
    --max_problems_to_sample_per_ruleset "$MAX_PROBLEMS_PER_RULESET" > "${LOG_DIR}/make_data.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: make_data_new.py failed. Check ${LOG_DIR}/make_data.log"
    exit 1
fi
echo "[Step 1] Completed. Log: ${LOG_DIR}/make_data.log"

# Step 2: Run downsample.py
INPUT_CSV="${SL_DIR}/simplelogic_heldout_k_sufficient_data_new.csv"
OUTPUT_CSV="${DATA_DIR}/simplelogic_heldout_k_sufficient_data_new_sampled.csv"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file $INPUT_CSV not found. make_data_new.py may have failed."
    exit 1
fi

echo "[Step 2] Running downsample.py..."
PYTHONPATH=. python SimpleLogic/downsample.py \
    --input_path "$INPUT_CSV" \
    --output_path "$OUTPUT_CSV" \
    --k1_sample "$K1_SAMPLE" \
    --k2_sample "$K2_SAMPLE" \
    --seed 42 \
    > "${LOG_DIR}/downsample.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: downsample.py failed. Check ${LOG_DIR}/downsample.log"
    exit 1
fi
echo "[Step 2] Completed. Log: ${LOG_DIR}/downsample.log"

# Step 3: Verify the final dataframe
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Error: Output file $OUTPUT_CSV not found. downsample.py may have failed."
    exit 1
fi

echo "[Step 3] Running verify_results.py..."
PYTHONPATH=. python SimpleLogic/tests/verify_results.py \
    --input_csv "$OUTPUT_CSV" \
    > "${LOG_DIR}/verify.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: verify_results.py failed. Check ${LOG_DIR}/verify.log"
    exit 1
fi
echo "[Step 3] Completed. Log: ${LOG_DIR}/verify.log"

echo "=========================================="
echo "All steps completed successfully!"
echo "Output files:"
echo "  - ${SL_DIR}/simplelogic_heldout_k_sufficient_prompts_new.csv"
echo "  - ${SL_DIR}/simplelogic_heldout_k_sufficient_data_new.csv"
echo "  - ${DATA_DIR}/simplelogic_heldout_k_sufficient_data_new_sampled.csv"
echo "Verification log: ${LOG_DIR}/verify.log"
echo "=========================================="
