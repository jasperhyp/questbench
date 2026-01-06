# Generate ruleset
For testing purpose only (only 0-10 instead of 0-7000):
```bash
PYTHONPATH=. python SimpleLogic/generate_ruleset_new.py \
      --sl_dir /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP \
      --start_idx 0 \
      --end_idx 10 \
      --max_k 4
```

Or `/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_{xxx}`.

Request a 72-cpu node in `test` for 12 hrs: `salloc -p test --account kempner_mzitnik_lab -t 0-12:00 --mem 960G -c 72`, then `reasoning_singularity` and run `bash external/questbench/scripts/run_logic_gen_parallel.sh`.

To run SimpleLogic rule sets other than the number 11 as questbench used, run `bash external/questbench/scripts/run_logic_gen_parallel_more_props.sh`

# Make data
```bash
PYTHONPATH=. python SimpleLogic/make_data_new.py \
      --sl_dir /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_{xxx} \
      --max_problems_to_sample_per_ruleset 50
```

# Evaluate
```bash
python mc_eval.py \
--model_name <model_name> \
--domain_name SL \
--eval_mode mc \
--data_dir /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_{xxx} \
--data_file /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_{xxx}/simplelogic_heldout_k_sufficient_data_sampled.csv \
--results_dir ./results \
--batch_size 64 \
(--use_invalid_facts_sets)
(--model_role_name assistant)
(--vllm_port <port>)
```

model_name: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8

eval_mode: mc|isambig|fullinfo|ask_k  # To be updated

prompt_mode: ""|"cot"|"fs4"  # To be updated

