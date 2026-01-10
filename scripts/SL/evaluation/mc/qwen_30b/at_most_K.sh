#!/bin/bash
#SBATCH --job-name=qwen-30b_logicq_mc_at-most-K
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# --- 1. Configuration ---
PROJECT_DIR="/n/home09/yeh803/workspace/Reasoning/external/questbench"
SINGULARITY_IMAGE="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/vllm_gptoss.sif"
# Use a fixed port to avoid conflicts with config files
PORT=8011
HOST="127.0.0.1"
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

# Create logs directory if it doesn't exist
mkdir -p ${PROJECT_DIR}/logs

echo "Job started on $(hostname) at $(date)"
echo "Using port: $PORT"

# --- 2. Run everything inside singularity ---
cd ${PROJECT_DIR}

singularity exec --nv --cleanenv \
    -B "$PWD":"$PWD" ${SINGULARITY_IMAGE} \
    bash --noprofile --norc -c "
        # Environment setup
        export CONDA_AUTO_ACTIVATE_BASE=false
        unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER
        
        source ~/.bashrc_singularity
        source .venv_vllm/bin/activate
        
        # Increase file descriptor limit
        ulimit -n 65535 2>/dev/null || true
        
        echo 'Starting vLLM server...'
        
        # Start vLLM in background
        CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL} \
            --reasoning-parser deepseek_r1 \
            --host ${HOST} \
            --port ${PORT} \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.95 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-num-batched-tokens 65536 \
            --max-num-seqs 64 \
            --kv-cache-dtype fp8 \
            > ${PROJECT_DIR}/logs_evaluation/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
        
        SERVER_PID=\$!
        
        # Cleanup on exit
        trap \"echo 'Stopping vLLM server...'; kill \$SERVER_PID 2>/dev/null\" EXIT
        
        # Wait for server to be ready
        echo 'Waiting for vLLM to load Qwen3-30B...'
        MAX_RETRIES=60
        COUNTER=0
        
        while ! curl -s http://${HOST}:${PORT}/health > /dev/null 2>&1; do
            if [ \$COUNTER -ge \$MAX_RETRIES ]; then
                echo 'Error: Server failed to start. Check logs/vllm_server_${SLURM_JOB_ID}.log'
                exit 1
            fi
            sleep 5
            ((COUNTER++))
            echo -n '.'
        done
        echo -e '\nServer is up and running!'
        
        # Set API environment variables for the Python scripts
        export OPENAI_API_KEY='EMPTY'
        export OPENAI_API_BASE='http://${HOST}:${PORT}/v1'

        # Run mc scripts
        echo '=== Starting mc scripts ==='
        cd ${PROJECT_DIR}
        
        echo 'Starting at_most_K config...'
        /n/home09/yeh803/workspace/Reasoning/.venv_vllm/bin/python mc_eval.py \
            --model_name Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
            --domain_name SL \
            --eval_mode mc \
            --data_dir /n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data/new_11_500k \
            --data_file /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/simplelogic_heldout_k_sufficient_data_new_sampled.csv \
            --results_dir ./results \
            --prompt_mode at_most_K \
            --batch_size 64 \
            > ${PROJECT_DIR}/logs_evaluation/qwen-at-most-K_${SLURM_JOB_ID}.log 2>&1

        EXIT_1=\$?
        echo 'Script finished with exit code: '\$EXIT_1
        echo 'Scripts completed at '\$(date)
        
        # Exit with error if either script failed
        if [ \$EXIT_1 -ne 0 ]; then
            echo 'Script failed!'
            exit 1
        fi
    "

echo "Job finished at $(date)"
