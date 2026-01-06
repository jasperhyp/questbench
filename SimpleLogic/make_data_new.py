# Copyright 2025 Google LLC
#
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

"""Format prompts and data for Logic-Q task."""

import argparse
import glob
import json
import orjson
import os
import random
import ast

import pandas as pd
from SimpleLogic import ruleset
import tqdm

tqdm = tqdm.tqdm

def main(arguments) -> None:
  # Load constructed rulesets
  rulesets = []
  for item_file in glob.glob(
      os.path.join(arguments.sl_dir, "*_heldout_fixed_new.jsonl")
  ):
    print(item_file)
    # with open(item_file, "r") as f:
    #   for line in tqdm(f):
    #     try:
    #       rulesets.append(json.loads(line))
    #     except json.JSONDecodeError:
    #       continue
    import gc
    gc.disable()
    
    with open(item_file, "rb") as f:
      for line in tqdm(f):
        try:
          rulesets.append(orjson.loads(line))
        except orjson.JSONDecodeError:
          continue
    
    gc.enable()

  # Create dataframe
  data = pd.DataFrame(
      columns=[
          "k",
          "known_facts",
          "known_untrue_facts",
          "cannot_ask_facts",
          "cannot_ask_facts_sets",
          "goal",
          "rules",
          "max_depth",
          "min_num_rules_needed",
          "num_constraints",
          "num_vars",
          "all_qs",
          "all_valid_qs",
          "gt_qs",
          # "gt_q_to_true_derivation",
          # "gt_q_to_false_derivation",
          "gt_q_to_derivations_min_rules",
          "gt_q_to_derivations_min_depth",
      ]
  )

  print(f"Processing {len(rulesets)} rulesets...")

  for rs in tqdm(rulesets):
    if "true_derivations" not in rs: 
        continue
        
    rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
    target_attr = rs["query"]
    
    k_data_source = {}
    if "heldout_k_sets" in rs and rs["heldout_k_sets"]:
      # Use the new structure: {"1": {ctx: [[v]]}, "2": {ctx: [[v1,v2]]}}
      k_data_source = rs["heldout_k_sets"]
    else:
      raise Exception

    # 2. Flatten all problems in this ruleset for subsampling
    # tuple: (k_int, context_str, valid_sets_list, derivations_min_rules_list, derivations_min_depth_list)
    all_problems = []
    for k_str, context_map in k_data_source.items():
      for context_str, dict_lst in context_map.items():
        all_problems.append((int(k_str), context_str, [dct["s_set"] for dct in dict_lst], [dct["derivations_min_rules"] for dct in dict_lst], [dct["derivations_min_depth"] for dct in dict_lst]))

    # 3. Subsample problems per ruleset (Robustness from original make_data)
    if len(all_problems) > arguments.max_problems_to_sample_per_ruleset:
      sampled_problems = random.sample(
          all_problems, arguments.max_problems_to_sample_per_ruleset
      )
    else:
      sampled_problems = all_problems

    # 4. Process each problem
    for k, context_str, valid_sets, derivations_min_rules, derivations_min_depth in sampled_problems:
      # Parse context
      # try:
      context_set = set(json.loads(context_str))
      # except:
      #     # Fallback if context_str is somehow malformed (e.g. single quotes)
      #     try:
      #       known_facts = ast.literal_eval(context_str)
      #     except:
      #       continue
      
      # Determine facts known to be false based on "not X" in context
      # NOTE: known facts are positive probably because simplelogic is built upon definite clauses restricted to positive literals
      false_facts = []
      for f in context_set:
        if f.startswith("not "):
            false_facts.append(f.split("not ")[1])
      false_facts = sorted(false_facts)
      true_facts = sorted([f for f in context_set if not f.startswith("not ")])
      
      # invalid questions and question sets
      # 1. Cannot ask the goal itself
      invalid_qs = {target_attr}
      # 2. Cannot ask about variables already in the known facts (redundant)
      known_vars = {f.split("not ")[-1] for f in context_set}
      invalid_qs.update(known_vars)
      
      invalid_q_sets = rs["context_to_invalid_sets"][str(k)].get(context_str, list())
      invalid_q_sets = {frozenset(q) for q in invalid_q_sets}
      
      # All variables in the tree (potential search space)
      # Get all variables that exist in both positive and negative forms
      all_vars = set()
      for q in rule_tree.nodes.keys():
        if q.startswith("not "):
          all_vars.add(q[4:])  # Extract variable name from "not X"
        else:
          all_vars.add(q)
      
      # Only include variables that have both positive and negative forms
      all_qs_atoms = set()
      for var in all_vars:
        if var in rule_tree.nodes and f"not {var}" in rule_tree.nodes:
          all_qs_atoms.add(var)
      
      # Valid individual variables available for selection (the atoms)
      valid_qs_atoms = sorted(list(
          all_qs_atoms - set(false_facts) - set(true_facts)
      ))
      
      # filter out sets that accidentally are identical to invalid sets.
      # TODO: Maybe this filter is not needed?
      clean_gt_qs = []
      clean_derivations_min_rules = []
      clean_derivations_min_depth = []
      num_rules_to_compute_q = []
      max_depth_to_compute_q = []
      for q_set, derivs_min_rules, derivs_min_depth in zip(valid_sets, derivations_min_rules, derivations_min_depth):
          if (not frozenset(q_set) in invalid_q_sets) and (not set(q_set).intersection(invalid_qs)):
              clean_gt_qs.append(sorted(q_set))
              clean_derivations_min_rules.append(derivs_min_rules)  # derivs is a dict of traces from all feasible combinations of true/false assignments
              clean_derivations_min_depth.append(derivs_min_depth)
              # NOTE: derivs_min_rules/min_depth is a dict where keys are assignment combinations and values are dicts with "derivation" key containing derive_obj.serialize()
              num_rules_to_compute_q.append(max(len(deriv["derivation"]["derivation"]) for deriv in derivs_min_rules.values()))
              max_depth_to_compute_q.append(max(max(deriv["derivation"]["leaf_words"].values()) for deriv in derivs_min_depth.values()))
      
      if not clean_gt_qs:
          continue
      
      num_rules = min(num_rules_to_compute_q)
      depth = min(max_depth_to_compute_q)
      num_total_rules = rule_tree.num_rules()
      num_words = rule_tree.num_words()

      data.loc[len(data)] = [
          k,
          true_facts,
          false_facts,
          sorted(list(invalid_qs)),
          [sorted(q_set) for q_set in invalid_q_sets],
          target_attr,
          rule_tree.serialize(),
          depth,
          num_rules,
          num_total_rules,
          num_words,
          sorted(list(all_qs_atoms)),
          valid_qs_atoms,
          clean_gt_qs,          # The main target for k-sufficient
          clean_derivations_min_rules,   # List of dicts of derivations for each gt_q
          clean_derivations_min_depth,   # List of dicts of derivations for each gt_q
      ]

  # Split into prompts and data
  if len(data) > 0:
      # Sample 25 prompts for few-shot usage
      prompt_indices = random.sample(range(len(data)), min(25, len(data)))
      prompts = data.iloc[prompt_indices]
      data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
      
      # Use a new filename to avoid overwriting 1-sufficient benchmarks
      prompt_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_prompts_new.csv")
      data_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_data_new.csv")
      
      print(f"Writing {len(prompts)} prompts to {prompt_path}")
      with open(prompt_path, "w") as f:
        prompts.to_csv(f, index=False)
        
      print(f"Writing {len(data_subsample)} rows to {data_path}")
      with open(data_path, "w") as f:
        data_subsample.to_csv(f, index=False)
  else:
      print("No valid data rows generated. Check input files or sampling logic.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--sl_dir",
      default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP",
      help="Directory containing the SimpleLogic data.",
  )
  parser.add_argument(
      "--max_problems_to_sample_per_ruleset",
      default=50,
      type=int,
      help="Maximum number of problems to sample per ruleset.",
  )
  main(parser.parse_args())
