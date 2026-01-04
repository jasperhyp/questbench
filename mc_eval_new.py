# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to evaluate LLMs on QuestBench domains."""

import argparse
import os
from evaluators.gsm import GSMEvaluator
from evaluators.gsm_new import GSMEvaluator as GSMEvaluator_new
from evaluators.planning import PlanningEvaluator
from evaluators.simple_logic import SimpleLogicEvaluator
import pandas as pd
import json
import time
import platform
import statistics
from datetime import datetime
import re

def main(user_args) -> None:
    domain_main_name = user_args.domain_name.split("_")[0]
    use_cot = False
    fs_samples = 0
    use_phys_constraints = False
    if user_args.prompt_mode == "cot":
        use_cot = True
    elif user_args.prompt_mode == "phys":
        use_phys_constraints = True
    elif user_args.prompt_mode.startswith("fs"):
        fs_samples = int(user_args.prompt_mode[2:])

    # Make directories for results and cache
    os.makedirs(user_args.results_dir, exist_ok=True)
    cache_dir = os.path.join(user_args.results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    data_file_base_name = os.path.splitext(os.path.basename(user_args.data_file))[0]

    # Make model name filesystem-safe (important: "Qwen/xxx" contains "/")
    safe_model_name = user_args.model_name.replace("/", "_")
    
    k_tag = "givek" if user_args.reveal_k_in_prompt else "nok"
    output_file_name = (
        f"{safe_model_name}-{user_args.domain_name}-{user_args.eval_mode}-"
        f"{user_args.prompt_mode}-{k_tag}-{data_file_base_name}"
    )

    cache_file = os.path.join(cache_dir, f"{output_file_name}.jsonl")
    output_file = os.path.join(user_args.results_dir, f"{output_file_name}.csv")

    # Ensure parent dirs exist (defensive)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("Loading Evaluator")

    if domain_main_name == "SL":
        evaluator = SimpleLogicEvaluator(
            user_args.model_name,
            cache_file=cache_file,
            use_cot=use_cot,
            fs_samples=fs_samples,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            model_role_name=user_args.model_role_name,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
        )
        prompt_file = os.path.join(
            user_args.data_dir,
            "questbench/questbench_data/Logic-Q/simplelogic_heldout_1k_prompts.csv",
        )
    elif domain_main_name == "GSM":
        if user_args.domain_name.split("_")[1] in ["csp", "verbal"] or user_args.k == 1:
            evaluator = GSMEvaluator(
                user_args.model_name,
                cache_file=cache_file,
                use_cot=use_cot,
                fs_samples=fs_samples,
                verbal_questions="verbal" in user_args.domain_name,
                eval_mode=user_args.eval_mode,
                batch_size=user_args.batch_size,
                model_role_name=user_args.model_role_name,
                parallel_model_calls=user_args.parallel_model_calls,
                vllm_port=user_args.vllm_port,
                reveal_k_in_prompt=user_args.reveal_k_in_prompt,
            )
            if user_args.domain_name.split("_")[1] == "verbal":
                prompt_file = os.path.join(
                    user_args.data_dir,
                    "questbench/questbench_data/GSM-Q/gsm_verbal_heldout_pilot_prompts.csv",
                )
            # if user_args.domain_name.split("_")[1] == "csp":
            else:
                prompt_file = os.path.join(
                    user_args.data_dir,
                    "questbench/questbench_data/GSM-Q/gsm_CSP_heldout_pilot_prompts.csv",
                )
            
        else:
            evaluator = GSMEvaluator_new(
                user_args.model_name,
                cache_file=cache_file,
                use_cot=use_cot,
                fs_samples=fs_samples,
                verbal_questions=False,
                eval_mode=user_args.eval_mode,
                batch_size=user_args.batch_size,
                model_role_name=user_args.model_role_name,
                parallel_model_calls=user_args.parallel_model_calls,
                vllm_port=user_args.vllm_port,
                reveal_k_in_prompt=user_args.reveal_k_in_prompt,
            )
            prompt_file = os.path.join(
                user_args.data_dir,
                f"gsm_new/gsm_CSP_heldout_pilot_prompts_missing_{user_args.k}.csv",
            )
            
    elif domain_main_name == "Planning":
        evaluator = PlanningEvaluator(
            user_args.model_name,
            domain_file=os.path.join(
                user_args.data_dir,
                "questbench/questbench_data/Planning-Q/task_pddls/blocks/domain.pddl",
            ),
            task_file_pattern=os.path.join(
                user_args.data_dir,
                "questbench/questbench_data/Planning-Q/task_pddls/blocks/task*.pddl",
            ),
            cache_file=cache_file,
            use_cot=use_cot,
            use_phys_constraints=use_phys_constraints,
            fs_samples=fs_samples,
            eval_mode=user_args.eval_mode,
            batch_size=user_args.batch_size,
            model_role_name=user_args.model_role_name,
            parallel_model_calls=user_args.parallel_model_calls,
            vllm_port=user_args.vllm_port,
        )
        prompt_file = os.path.join(
            user_args.data_dir,
            "questbench/questbench_data/Planning-Q/planning_heldout_prompts.csv",
        )
    else:
        raise SystemExit(f"Unknown domain: {domain_main_name}")

    print("Loading Data")
    data_file = user_args.data_file
    
    if user_args.k == 1:
        data_file = 'questbench/questbench_data/GSM-Q/gsm_CSP_full.csv'
    data_file = os.path.join(
            user_args.data_dir,
            data_file,
        )
    with open(data_file, "r") as f:
        data = pd.read_csv(f)
    prompt_data = None
    if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                prompt_data = pd.read_csv(f)


    print("Starting Evaluation")
    out = evaluator.evaluate_data(data, prompt_data)
    if isinstance(out, tuple):
        results, all_cots, total_cost = out
    else:
        results = out
        all_cots = None
        total_cost = None

    with open(output_file, "w") as wf:
        results.to_csv(wf)
    print(f"Wrote to {output_file}")
    
    if total_cost is not None:
        with open(output_file.replace(".csv", "_cost.json"), "w") as wf:
            json.dump(total_cost, wf)
            
        # -------------------------
    # Write evaluation summary
    # -------------------------
    summary_file = output_file.replace(".csv", "_summary.json")

    def _to_float_list(x):
        # total_cost could be list[float], dict, or None
        if x is None:
            return []
        if isinstance(x, list):
            vals = []
            for v in x:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return vals
        if isinstance(x, dict):
            vals = []
            for v in x.values():
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return vals
        # fallback scalar
        try:
            return [float(x)]
        except Exception:
            return []

    def _safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    def _safe_median(xs):
        try:
            return float(statistics.median(xs)) if xs else None
        except Exception:
            return None

    def _safe_min(xs):
        return float(min(xs)) if xs else None

    def _safe_max(xs):
        return float(max(xs)) if xs else None

    # basic counts
    num_rows = int(len(results)) if results is not None else 0
    num_scored = int(results["correct"].notna().sum()) if (results is not None and "correct" in results.columns) else 0

    # accuracy
    acc = None
    if results is not None and num_scored > 0:
        try:
            acc = float(results.loc[results["correct"].notna(), "correct"].mean())
        except Exception:
            acc = None

    # cost stats
    cost_list = _to_float_list(total_cost)
    cost_stats = {
        "total_cost": float(sum(cost_list)) if cost_list else None,
        "avg_cost": _safe_mean(cost_list),
        "median_cost": _safe_median(cost_list),
        "min_cost": _safe_min(cost_list),
        "max_cost": _safe_max(cost_list),
        "num_cost_entries": int(len(cost_list)),
    }

    # per-k accuracy
    acc_by_k = {}
    if results is not None and "k" in results.columns and num_scored > 0:
        try:
            tmp = results.loc[results["correct"].notna()].groupby("k")["correct"].mean()
            acc_by_k = {str(int(k)): float(v) for k, v in tmp.to_dict().items()}
        except Exception:
            acc_by_k = {}

    # per-depth accuracy
    acc_by_depth = {}
    depth_col = None
    for c in ["max_depth", "depth"]:
        if results is not None and c in results.columns:
            depth_col = c
            break
    if results is not None and depth_col is not None and num_scored > 0:
        try:
            tmp = results.loc[results["correct"].notna()].groupby(depth_col)["correct"].mean()
            # depth may be NaN
            acc_by_depth = {str(k): float(v) for k, v in tmp.to_dict().items()}
        except Exception:
            acc_by_depth = {}

    # confusion style counts
    correct_count = None
    incorrect_count = None
    if results is not None and num_scored > 0:
        try:
            correct_count = int(results.loc[results["correct"] == True].shape[0])
            incorrect_count = int(results.loc[results["correct"] == False].shape[0])
        except Exception:
            pass

    # label cardinality stats for mc only
    pred_set_sizes = []
    if user_args.eval_mode == "mc" and results is not None and "pred_q" in results.columns:
        for x in results["pred_q"].tolist():
            if isinstance(x, (set, list, tuple)):
                pred_set_sizes.append(len(x))
            elif isinstance(x, str):
                nums = re.findall(r"\b[0-9]+\b", x)
                if nums:
                    pred_set_sizes.append(len(set(nums)))

    pred_set_stats = {
        "avg_pred_set_size": _safe_mean(pred_set_sizes) if pred_set_sizes else None,
        "median_pred_set_size": _safe_median(pred_set_sizes) if pred_set_sizes else None,
        "min_pred_set_size": _safe_min(pred_set_sizes) if pred_set_sizes else None,
        "max_pred_set_size": _safe_max(pred_set_sizes) if pred_set_sizes else None,
        "num_pred_set_entries": int(len(pred_set_sizes)),
    }

    # run meta
    summary = {
        "run": {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "config": {
            "model_name": user_args.model_name,
            "domain_name": user_args.domain_name,
            "eval_mode": user_args.eval_mode,
            "prompt_mode": user_args.prompt_mode,
            "batch_size": user_args.batch_size,
            "model_role_name": user_args.model_role_name,
            "parallel_model_calls": bool(user_args.parallel_model_calls),
            "vllm_port": user_args.vllm_port,
            "k_arg": user_args.k,
            "reveal_k_in_prompt": bool(user_args.reveal_k_in_prompt),  # NEW
        },
        "paths": {
            "data_file": data_file,
            "prompt_file": prompt_file,
            "cache_file": cache_file,
            "results_csv": output_file,
            "cots_json": output_file.replace(".csv", "_cots.json") if all_cots is not None else None,
            "cost_json": output_file.replace(".csv", "_cost.json") if total_cost is not None else None,
            "summary_json": summary_file,
        },
        "data": {
            "num_rows_total": num_rows,
            "num_rows_scored": num_scored,
        },
        "metrics": {
            "accuracy": acc,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy_by_k": acc_by_k,
            "accuracy_by_depth": acc_by_depth,
            "cost": cost_stats,
            "pred_set_size": pred_set_stats,
        },
    }

    with open(summary_file, "w") as wf:
        json.dump(summary, wf, indent=2)
    print(f"Wrote summary to {summary_file}")
        


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_name",
      type=str,
      default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
      help=(
          "The name of the model to evaluate. Currently support `gpt-4o`,"
          " `o1-preview`, `gemini-1.5-flash`, `gemini-1.5-pro`, `gemma_2_2b`,"
          " `gemma_2_9b`, and `gemma_2_27b`"
      ),
  )
  parser.add_argument(
      "--domain_name",
      type=str,
      choices=[
          "SL",
          "GSM_csp",
          "GSM_verbal",
          "GSM_k",
          "Planning",
      ],
      help=(
          "Domain name. `SL` is for Simple Logic, `GSM_csp` is for GSM-Q with"
          " CSPs, `GSM_verbal` is for GSM-Q with verbal questions, and"
          " `Planning` is for Planning-Q."
      ),
  )
  parser.add_argument(
      "--eval_mode",
      type=str,
      choices=[
          "mc",
          "sc",        # single-choice missing-count
          "isambig",
          "fullinfo",
      ],
      help=(
          "Evaluation mode. `mc` is for the multiple choice version of"
          " QuestBench (select questions). `sc` is for single-choice where the"
          " model selects how many required variables are missing (0-4)."
          " `isambig` is for evaluating whether the model can"
          " identify the task is ambiguous, and `fullinfo` is for evaluating"
          " the model's performance on the task with the full information"
          " (i.e., no missing information)."
      ),
  )
  parser.add_argument(
      "--data_file", type=str, help="The path to the data file.", default=None
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="questbench_data",
      help=(
          "Directory containing data. Default is `questbench_data` in the"
          " current directory."
      ),
  )
  parser.add_argument(
      "--prompt_mode",
      type=str,
      choices=["", "cot", "fs4"],
      default="",
      help="Use vanilla, CoT, or fewshot prompting (with 4 samples).",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="results",
      help=(
          "Directory to write results to. Default is `results` in the current"
          " directory."
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=1,
      help="Batch size for evaluation.",
  )
  parser.add_argument(
      "--model_role_name",
      type=str,
      default="model",
      help=(
          "The name of the model role. In Gemini, this should be `model`. In"
          " OpenAI, this should be `assistant`. You can use other role names as"
          " needed."
      ),
  )
  parser.add_argument(
      "--no_thread_pool",
      action="store_false",
      dest="parallel_model_calls",
      help="Disable thread pool.",
  )
  parser.add_argument(
      "--vllm_port",
      type=int,
      default=8000,
      help="Port for the VLLM server. Default is 8000.",
  )
  parser.add_argument(
      "--k",
      type=int,
      default=2,
      help="k-sufficient.",
  )
  
  parser.add_argument(
        "--reveal_k_in_prompt",
        action="store_true",
        help="If set, include the exact k in the MC system prompt.",
  )
  
  args = parser.parse_args()
  main(args)
