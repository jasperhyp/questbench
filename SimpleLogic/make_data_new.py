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
import os
import random
import ast

import pandas as pd
# from SimpleLogic import holdout_utils_new
from SimpleLogic import ruleset
import tqdm

tqdm = tqdm.tqdm


def main(arguments) -> None:
  # Load constructed rulesets
  rulesets = []
  for item_file in glob.glob(
      os.path.join(arguments.sl_dir, "*_heldout_fixed_new_test.jsonl")
  ):
    print(item_file)
    with open(item_file, "r") as f:
      for line in tqdm(f):
        try:
          rulesets.append(json.loads(line))
        except json.JSONDecodeError:
          continue

  # Create dataframe
  data = pd.DataFrame(
      columns=[
          "k",
          "known_facts",
          "known_untrue_facts",
          "cannot_ask_facts",
          "goal",
          "rules",
          "max_depth",
          "min_num_rules_needed",
          "num_constraints",
          "num_vars",
          "all_qs",
          "all_valid_qs",
          "gt_qs",
          "gt_q_to_true_derivation",
          "gt_q_to_false_derivation",
      ]
  )

  # for rs in tqdm(rulesets):
  #   if ruleset.get("heldout_set_to_q", []):
  #     rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
  #     target_attr = rs["query"]
  #     heldout_set_to_q = rs["heldout_set_to_q"]
  #     heldout_set_to_subset_qs = rs["heldout_set_to_subset_qs"]

  #     heldout_set_to_q_subsample = random.sample(
  #         list(heldout_set_to_q.keys()),
  #         min(
  #             len(heldout_set_to_q),
  #             arguments.max_problems_to_sample_per_ruleset,
  #         ),
  #     )

  #     for heldout_prompt in heldout_set_to_q_subsample:
  #       gt_qs = set()
  #       invalid_qs = set()  # makes problem easier; request LM not to ask
  #       false_facts = []
  #       num_rules_to_compute_q = []
  #       max_depth_to_compute_q = []
  #       gt_q_to_true_derivation = {}
  #       gt_q_to_false_derivation = {}
  #       # FIXME
  #       # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  #       # for differ_word in heldout_set_to_q[heldout_prompt]:
  #       #   gt_qs.add(differ_word.split("not ")[-1])
  #       #   true_derivation, false_derivation = (
  #       #       heldout_set_to_q[heldout_prompt][differ_word]["true_derivation"],
  #       #       heldout_set_to_q[heldout_prompt][differ_word]["false_derivation"],
  #       #   )
  #       #   gt_q_to_true_derivation[differ_word] = true_derivation["derivation"]
  #       #   gt_q_to_false_derivation[differ_word] = false_derivation["derivation"]
  #       # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  #       target_vars_sets = []
  #       for differ_key in heldout_set_to_q[heldout_prompt]:
  #         vars_in_key = differ_key.split(" & ")
  #         clean_vars = tuple(sorted([v.split("not ")[-1] for v in vars_in_key]))
  #         target_vars_sets.append(clean_vars)
  #         tuple(sorted([v.split("not ")[-1] for v in vars_in_key]))
  #         target_vars_sets.append(clean_vars)
          
  #         true_derivation, false_derivation = (
  #             heldout_set_to_q[heldout_prompt][differ_key]["true_derivation"],
  #             heldout_set_to_q[heldout_prompt][differ_key]["false_derivation"],
  #         )
  #         gt_q_to_true_derivation[differ_key] = true_derivation["derivation"]
  #         gt_q_to_false_derivation[differ_key] = false_derivation["derivation"]
  #       # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  #         num_rules_to_compute_q.append(
  #             max(
  #                 len(true_derivation["derivation"]),
  #                 len(false_derivation["derivation"]),
  #             )
  #         )
  #         max_depth_to_compute_q.append(
  #             max(
  #                 max(true_derivation["leaf_words"].values()),
  #                 max(false_derivation["leaf_words"].values()),
  #             )
  #         )
  #         for cannot_ask_attr in set(heldout_set_to_subset_qs[heldout_prompt]):
  #           invalid_qs.add(cannot_ask_attr)

  #       if set(false_facts).intersection(set(invalid_qs)):
  #         # remove invalid_qs
  #         false_facts = list(set(false_facts).difference(set(invalid_qs)))
  #         # sort
  #         false_facts = sorted(false_facts)

  #       invalid_qs = sorted(list(invalid_qs))
  #       if set(gt_qs) <= set(invalid_qs):
  #         continue

  #       all_qs = set(
  #           {q for q in rule_tree.nodes.keys() if not q.startswith("not ")}
  #       )
  #       valid_qs = (
  #           set(all_qs)
  #           - set(invalid_qs)
  #           - set(false_facts)
  #           - set(heldout_prompt)
  #       )

  #       # check
  #       # FIXME
  #       # Update Validation Logic
  #       # The original code loops through `all_qs` (single variables) and checks if they solve the problem.
  #       # For k=2, NO single variable will solve it. We must check the combinations.
  #       # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  #       true_facts = json.loads(heldout_prompt)
  #       inferrable_facts = holdout_utils_new.get_all_inferrable_facts(
  #           rule_tree, true_facts, false_facts
  #       )
  #       try:
  #         assert (
  #             target_attr not in inferrable_facts
  #             and ruleset.negate(target_attr) not in inferrable_facts
  #         )
  #       except AssertionError:
  #         continue
  #       new_gt_qs = set()
  #       for q in all_qs:
  #         if q == target_attr:
  #           continue
  #         inferrable_q_facts = holdout_utils_new.get_all_inferrable_facts(
  #             rule_tree, true_facts + [q], false_facts
  #         )
  #         inferrable_negq_facts = holdout_utils_new.get_all_inferrable_facts(
  #             rule_tree, true_facts, false_facts + [ruleset.negate(q)]
  #         )
  #         if q in gt_qs:
  #           assert (
  #               target_attr in inferrable_q_facts
  #               or ruleset.negate(target_attr) in inferrable_q_facts
  #           )
  #           if target_attr in inferrable_q_facts:
  #             if ruleset.negate(target_attr) in inferrable_negq_facts:
  #               new_gt_qs.add(q)
  #           else:
  #             if target_attr not in inferrable_negq_facts:
  #               new_gt_qs.add(q)
  #         elif q not in invalid_qs:
  #           if not (
  #               (
  #                   target_attr not in inferrable_q_facts
  #                   and ruleset.negate(target_attr) not in inferrable_q_facts
  #               )
  #               or (
  #                   target_attr not in inferrable_negq_facts
  #                   and ruleset.negate(target_attr) not in inferrable_negq_facts
  #               )
  #           ):
  #             new_gt_qs.add(q)
  #       if not new_gt_qs:
  #         continue
  #       gt_qs = sorted(list(new_gt_qs))

  #       num_rules = min(num_rules_to_compute_q)
  #       depth = min(max_depth_to_compute_q)
  #       num_total_rules = rule_tree.num_rules()
  #       num_words = rule_tree.num_words()
  #       data.loc[len(data)] = [
  #           sorted(json.loads(heldout_prompt)),
  #           sorted(false_facts),
  #           sorted(invalid_qs),
  #           target_attr,
  #           rule_tree.serialize(),
  #           depth,
  #           num_rules,
  #           num_total_rules,
  #           num_words,
  #           all_qs,
  #           valid_qs,
  #           gt_qs,
  #           gt_q_to_true_derivation,
  #           gt_q_to_false_derivation,
  #       ]

  # # split into prompts and data
  # prompt_indices = random.sample(range(len(data)), 25)
  # prompts = data.iloc[prompt_indices]
  # data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
  # # save data
  # with open(
  #     os.path.join(arguments.sl_dir, "temp_simplelogic_heldout_1k_prompts.csv"), "w"
  # ) as f:
  #   prompts.to_csv(f, index=False)
  # with open(
  #     os.path.join(arguments.sl_dir, "temp_simplelogic_heldout_1k_data.csv"), "w"
  # ) as f:
  #   data_subsample.to_csv(f, index=False)
  
  print(f"Processing {len(rulesets)} rulesets...")

  for rs in tqdm(rulesets):
    if "rules" not in rs: 
        continue
        
    rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
    target_attr = rs["query"]
    
    # 1. Identify where the problem data is stored
    # We prefer 'heldout_k_sets' if it exists.
    k_data_source = {}
    
    if "heldout_k_sets" in rs:
      # Use the new structure: {"1": {ctx: [[v]]}, "2": {ctx: [[v1,v2]]}}
      k_data_source = rs["heldout_k_sets"]
    elif "heldout_set_to_q" in rs:
      # Fallback for old 1-sufficient only files
      # Convert {ctx: {v: info}} -> {"1": {ctx: [[v]]}}
      k_data_source = {"1": {}}
      for h_set, q_dict in rs["heldout_set_to_q"].items():
        k_data_source["1"][h_set] = [[v] for v in q_dict.keys()]
    else:
      continue

    # 2. Flatten all problems in this ruleset for subsampling
    # tuple: (k_int, context_str, valid_sets_list)
    all_problems = []
    for k_str, context_map in k_data_source.items():
      for context_str, valid_sets in context_map.items():
        all_problems.append((int(k_str), context_str, valid_sets))

    # 3. Subsample problems per ruleset
    if len(all_problems) > arguments.max_problems_to_sample_per_ruleset:
      sampled_problems = random.sample(
          all_problems, arguments.max_problems_to_sample_per_ruleset
      )
    else:
      sampled_problems = all_problems

    # 4. Process each problem
    for k, context_str, valid_sets in sampled_problems:
      # Parse context
      try:
          known_facts = json.loads(context_str)
      except:
          # Fallback if context_str is somehow malformed (e.g. single quotes)
          try:
            known_facts = ast.literal_eval(context_str)
          except:
            continue
      
      # Determine facts known to be false based on "not X" in context
      false_facts = []
      for f in known_facts:
        if f.startswith("not "):
            false_facts.append(f.split("not ")[1])
      false_facts = sorted(false_facts)
      
      # Determine invalid questions
      # 1. Cannot ask the goal itself
      invalid_qs = {target_attr}
      # 2. Cannot ask about variables already in the known facts (redundant)
      known_vars = {f.split("not ")[-1] for f in known_facts}
      invalid_qs.update(known_vars)

      # Extract GT Derivations (Only available easily for k=1 via heldout_set_to_q)
      # For k>1, we leave these dicts empty.
      gt_q_to_true_derivation = {}
      gt_q_to_false_derivation = {}
      
      if k == 1 and "heldout_set_to_q" in rs and context_str in rs["heldout_set_to_q"]:
        q_info = rs["heldout_set_to_q"][context_str]
        # valid_sets is [[v1], [v2]...]; we look them up in the old dict
        for v_list in valid_sets:
            v = v_list[0] 
            if v in q_info:
                gt_q_to_true_derivation[v] = q_info[v]["true_derivation"]["derivation"]
                gt_q_to_false_derivation[v] = q_info[v]["false_derivation"]["derivation"]

      # valid_sets comes from JSON as list of lists, e.g. [['a', 'b'], ['c', 'd']]
      gt_qs = valid_sets
      
      # Basic Validation: Ensure GT sets don't overlap with invalid_qs
      clean_gt_qs = []
      for q_set in gt_qs:
          # If any variable in the solution set is already known/invalid, skip this solution
          if not set(q_set).intersection(invalid_qs):
              clean_gt_qs.append(sorted(q_set))
      
      if not clean_gt_qs:
          continue

      # All variables in the tree (potential search space)
      all_qs = set(
          {q for q in rule_tree.nodes.keys() if not q.startswith("not ")}
      )
      
      # Valid individual variables available for selection (the atoms)
      valid_qs_atoms = sorted(list(
          all_qs - invalid_qs - set(false_facts)
      ))

      # Stats
      num_rules = rs.get("depth", 0) 
      depth = rs.get("depth", 0)
      
      # Refine depth stats if we have derivation info (k=1 only)
      if gt_q_to_true_derivation:
          first_key = list(gt_q_to_true_derivation.keys())[0]
          # derivation is a list of rule steps
          num_rules = len(gt_q_to_true_derivation[first_key])
      
      num_total_rules = rule_tree.num_rules()
      num_words = rule_tree.num_words()

      data.loc[len(data)] = [
          k,
          sorted(known_facts),
          sorted(false_facts),
          sorted(list(invalid_qs)),
          target_attr,
          rule_tree.serialize(),
          depth,
          num_rules,
          num_total_rules,
          num_words,
          sorted(list(all_qs)),
          valid_qs_atoms,
          clean_gt_qs,          # The main target for k-sufficient
          gt_q_to_true_derivation,
          gt_q_to_false_derivation,
      ]

  # Split into prompts and data
  if len(data) > 0:
      # Sample 25 prompts for few-shot usage
      prompt_indices = random.sample(range(len(data)), min(25, len(data)))
      prompts = data.iloc[prompt_indices]
      data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
      
      # Use a new filename to avoid overwriting 1-sufficient benchmarks
      prompt_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_prompts_test.csv")
      data_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_data_test.csv")
      
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
