# Generate ruleset
For testing purpose only (only 0-10 instead of 0-7000):
```bash
PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
      --sl_dir /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP \
      --start_idx 0 \
      --end_idx 10 \
      --max_k 4
```
Typically, we request a 72-cpu node in `test` for 12 hrs: `salloc -p test --account kempner_mzitnik_lab -t 0-12:00 --mem 960G -c 72`, then (ruleset_idx is from 0 to 30, max_layer_prod is 500000/1000000):
```bash
 bash scripts/data_generation/run_logic_gen_parallel_more_props_for_oom.sh start_ruleset_idx end_ruleset_idx max_layer_prod num_parallel
```
E.g., 
```bash
bash scripts/data_generation/run_logic_gen_parallel_more_props_for_oom.sh 23 24 500000 18
```

# Make data, downsample, verify
```bash
bash scripts/data_generation/make_data_and_downsample.sh new_{ruleset_idx}_{500k\|1m} 20 2000 2000
```

# Evaluate
```bash
      CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 --reasoning-parser deepseek_r1 --host 0.0.0.0 --port 8011 --max-model-len 65536 --gpu-memory-utilization 0.95  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 65536 --max-num-seqs 32 --kv-cache-dtype fp8
```



```bash
python mc_eval.py \
--model_name <model_name> \
--domain_name SL \
--eval_mode mc \
--data_dir /n/netscratch/mzitnik_lab/Everyone/yeh803/Reasoning/logic_q_data/Logic-Q/RP/RP/new_{xxx}_{500k\|1m} \
--data_file /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_{xxx}_{500k\|1m}/simplelogic_heldout_k_sufficient_data_sampled.csv \
--results_dir ./results \
--prompt_mode {at_most_K\|exact_k} \
--batch_size 64 \
(--use_invalid_facts_sets)
(--model_role_name assistant)
(--vllm_port <port>)
```

eval_mode: mc|isambig|fullinfo|ask_k  # To be updated
