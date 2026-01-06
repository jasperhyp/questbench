python mc_eval.py \
--model_name Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
--domain_name SL \
--eval_mode mc \
--data_dir /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k \
--data_file /n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k/simplelogic_heldout_k_sufficient_data_new_sampled.csv \
--results_dir ./results \
--batch_size 64 \
(--use_invalid_facts_sets)
(--model_role_name assistant)
(--vllm_port <port>)