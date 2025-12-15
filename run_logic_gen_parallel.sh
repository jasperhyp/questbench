#!/bin/bash

# Define the data directory path to keep the command clean
SL_DIR="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP"

# Loop from 4000 to 6500 with increments of 500
# {START..END..STEP}
for start_idx in {4000..6500..500}; do
    
    # Calculate the end index (start + 500)
    end_idx=$((start_idx + 500))

    echo "Launching job: start_idx $start_idx -> end_idx $end_idx"

    # Run the command in the background (&)
    PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
        --sl_dir "$SL_DIR" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        --max_k 4 &

done

# Wait for all background jobs to finish before exiting the script
wait

echo "All jobs completed."
