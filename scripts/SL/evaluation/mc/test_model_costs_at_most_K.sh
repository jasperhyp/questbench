#!/bin/bash
#SBATCH --job-name=qwen-4b_logicq_mc_at-most-K
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=6:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# --- 1. Configuration ---
PROJECT_DIR="/n/home09/yeh803/workspace/Reasoning/external/questbench"
SINGULARITY_IMAGE="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/vllm_gptoss.sif"
# Use a fixed port to avoid conflicts with config files
PORT=8011
HOST="127.0.0.1"
MODEL_1="Qwen/Qwen3-4B-Thinking-2507"
MODEL_2="gpt-5"
MODEL_3="gemini-3-flash-preview"

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
        CUDA_VISIBLE_DEVICES=0 vllm serve ${MODEL_1} \
            --reasoning-parser deepseek_r1 \
            --host ${HOST} \
            --port ${PORT} \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.95 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-num-batched-tokens 65536 \
            --max-num-seqs 32 \
            > ${PROJECT_DIR}/logs_evaluation/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
        
        SERVER_PID=\$!
        
        # Cleanup on exit
        trap \"echo 'Stopping vLLM server...'; kill \$SERVER_PID 2>/dev/null\" EXIT
        
        # Wait for server to be ready
        echo 'Waiting for vLLM to load Qwen3-4B...'
        MAX_RETRIES=10
        COUNTER=0
        
        while ! curl -s http://${HOST}:${PORT}/health > /dev/null 2>&1; do
            if [ \$COUNTER -ge \$MAX_RETRIES ]; then
                echo 'Error: Server failed to start. Check logs_evaluation/vllm_server_${SLURM_JOB_ID}.log'
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

        echo 'Starting at_most_K config with GPT-5...'
        python mc_eval.py \
            --model_name ${MODEL_2} \
            --domain_name SL \
            --eval_mode mc \
            --data_dir /n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data/new_11_500k \
            --data_file /n/home09/yeh803/workspace/Reasoning/notebooks/test_costs.csv \
            --results_dir ./results/mc/at_most_K/gpt_5/ \
            --prompt_mode at_most_K \
            --batch_size 64 \
            > ${PROJECT_DIR}/logs_evaluation/gpt-5_logicq_mc_at-most-K_${SLURM_JOB_ID}.log 2>&1

        EXIT_2=\$?
        echo 'Script finished with exit code: '\$EXIT_2

        echo 'Starting at_most_K config with Gemini 3 Flash...'
        python mc_eval.py \
            --model_name ${MODEL_3} \
            --domain_name SL \
            --eval_mode mc \
            --data_dir /n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data/new_11_500k \
            --data_file /n/home09/yeh803/workspace/Reasoning/notebooks/test_costs.csv \
            --results_dir ./results/mc/at_most_K/gemini_3_flash/ \
            --prompt_mode at_most_K \
            --batch_size 64 \
            > ${PROJECT_DIR}/logs_evaluation/gemini-3-flash_logicq_mc_at-most-K_${SLURM_JOB_ID}.log 2>&1

        EXIT_3=\$?
        echo 'Script finished with exit code: '\$EXIT_3
        
        echo 'Starting at_most_K config with Qwen-4B...'
        python mc_eval.py \
            --model_name ${MODEL_1} \
            --domain_name SL \
            --eval_mode mc \
            --data_dir /n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data/new_11_500k \
            --data_file /n/home09/yeh803/workspace/Reasoning/notebooks/test_costs.csv \
            --results_dir ./results/mc/at_most_K/qwen_4b/ \
            --prompt_mode at_most_K \
            --batch_size 64 \
            > ${PROJECT_DIR}/logs_evaluation/qwen-4b_logicq_mc_at-most-K_${SLURM_JOB_ID}.log 2>&1

        EXIT_1=\$?
        echo 'Script finished with exit code: '\$EXIT_1

        echo 'Scripts completed at '\$(date)
        
        # Exit with error if either script failed
        if [ \$EXIT_1 -ne 0 ] || [ \$EXIT_2 -ne 0 ] || [ \$EXIT_3 -ne 0 ]; then
            echo 'Script failed!'
            exit 1
        fi
    "

echo "Job finished at $(date)"
