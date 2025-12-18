import argparse
import glob
import json
import os
import random
import ast
import itertools

import pandas as pd
from SimpleLogic import ruleset
from SimpleLogic import holdout_utils_new
import tqdm

tqdm = tqdm.tqdm


def find_matching_derivation(derivations_list, context_set, solution_vars):
    """
    Reconstructs the derivation trace by finding the best rule in the list
    whose leaf nodes are satisfied by the (Context U Solution).
    
    Prioritizes:
    1. Consistency (cannot require 'A' and 'not A')
    2. Simplicity (fewer rules/steps in derivation)
    """
    # Filter and sort candidates
    candidates = []
    
    for d in derivations_list:
        # d is the serialized derivation dictionary
        # d['leaf_words'] keys are the premises required
        # d['derivation'] is the trace list
        leaves = d.get('leaf_words', {}).keys()
        
        is_match = True
        required_vars_from_solution = set()
        
        # Check 1: Are all leaves available?
        for leaf in leaves:
            # A. Available in Context?
            if leaf in context_set:
                continue
            
            # B. Available in Solution (User Question)?
            # If leaf is "A" or "not A", and we ask about "A", we get the value.
            var_name = leaf.split("not ")[-1] if leaf.startswith("not ") else leaf
            if var_name in solution_vars:
                required_vars_from_solution.add(var_name)
                continue
                
            # If neither, we can't form this derivation
            is_match = False
            break
        
        if not is_match:
            continue
            
        # Check 2: Consistency
        # A derivation cannot require both 'A' and 'not A' as leaves.
        # (The context is assumed consistent, but we check the parts coming from solution)
        leaf_vars_map = {} # var -> polarity
        consistent = True
        for leaf in leaves:
            var = leaf.split("not ")[-1] if leaf.startswith("not ") else leaf
            polarity = "neg" if leaf.startswith("not ") else "pos"
            
            if var in leaf_vars_map:
                if leaf_vars_map[var] != polarity:
                    consistent = False
                    break
            leaf_vars_map[var] = polarity
            
        if not consistent:
            continue

        # Score the candidate: Length of derivation (primary), then number of solution vars used (secondary)
        trace = d['derivation']
        score = (len(trace), len(required_vars_from_solution))
        candidates.append((score, trace))

    # Sort to find the "simplest" valid derivation
    # Sorts by length ascending
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
        
    return None
  

def main(arguments) -> None:
  # Load constructed rulesets
  rulesets = []
  search_pattern = os.path.join(arguments.sl_dir, "*_heldout_fixed_new.jsonl")
  print(f"Searching for rulesets in: {search_pattern}")
  
  for item_file in glob.glob(search_pattern):
    print(f"Loading {item_file}")
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
  
  print(f"Processing {len(rulesets)} rulesets...")

  for rs in tqdm(rulesets):
    if "rules" not in rs: 
        continue
        
    rule_tree = ruleset.RuleTree.deserialize(rs["rules"])
    target_attr = rs["query"]
    
    # 1. Identify where the problem data is stored
    # We prefer 'heldout_k_sets' if it exists.
    k_data_source = {}
    if "heldout_k_sets" in rs and rs["heldout_k_sets"]:
      k_data_source = rs["heldout_k_sets"]
    elif "heldout_set_to_q" in rs:
      # Fallback for old 1-sufficient only files
      k_data_source = {"1": {}}
      for h_set, q_dict in rs["heldout_set_to_q"].items():
        k_data_source["1"][h_set] = [[v] for v in q_dict.keys()]
    else:
      continue

    # 2. Flatten all problems in this ruleset for subsampling
    all_problems = []
    for k_str, context_map in k_data_source.items():
      if not context_map: continue # Skip empty k levels
      for context_str, valid_sets in context_map.items():
        if not valid_sets: continue # Skip empty valid sets (e.g. "[]")
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
          try:
            known_facts = ast.literal_eval(context_str)
          except:
            continue
      
      context_set = set(known_facts)

      # Determine facts known to be false based on "not X" in context
      false_facts = []
      for f in known_facts:
        if f.startswith("not "):
            false_facts.append(f.split("not ")[1])
      false_facts = sorted(false_facts)
      
      # Determine invalid questions
      invalid_qs = {target_attr}
      known_vars = {f.split("not ")[-1] for f in known_facts}
      invalid_qs.update(known_vars)
      
      # --- DEFINING SEARCH SPACE ---
      all_qs = set(
          {q for q in rule_tree.nodes.keys() if not q.startswith("not ")}
      )
      search_space_vars = sorted(list(all_qs - invalid_qs))

      # --- VALIDATION CHECKS ---
      
      # Check 0: Context Insufficiency (Is it already solved?)
      if holdout_utils_new.check_sufficiency(rule_tree, context_set, set(), target_attr):
          continue

      # Check 1: k-Sufficiency
      clean_gt_qs = []
      for q_set in valid_sets:
          # Filter sets containing invalid vars
          if set(q_set).intersection(invalid_qs):
              continue
          
          # Verify sufficiency using holdout_utils
          if holdout_utils_new.check_sufficiency(rule_tree, context_set, set(q_set), target_attr):
              clean_gt_qs.append(sorted(q_set))
      
      if not clean_gt_qs:
          continue

      # Check 2: Global Minimality (for k > 1)
      # Ensure no smaller set exists that solves the problem
      is_minimal = True
      if k > 1:
          for size in range(1, k):
              found_smaller = False
              # Brute force check all subsets of size < k
              # Limit search space if too large? Usually |search_space| ~20
              for subset in itertools.combinations(search_space_vars, size):
                  if holdout_utils_new.check_sufficiency(rule_tree, context_set, set(subset), target_attr):
                      found_smaller = True
                      break
              if found_smaller:
                  is_minimal = False
                  break
      
      if not is_minimal:
          continue
      
      # --- TRACE RECONSTRUCTION ---
      gt_q_to_true_derivation = {}
      gt_q_to_false_derivation = {}
      
      for sol_list in clean_gt_qs:
          solution_vars = set(sol_list)
          
          # Key Format
          if len(sol_list) == 1:
              sol_key = sol_list[0]
          else:
              sol_key = str(tuple(sorted(sol_list)))

          # Find best Proof for True
          true_trace = find_matching_derivation(
              rs.get("true_derivations", []), 
              context_set, 
              solution_vars
          )
          if true_trace:
              gt_q_to_true_derivation[sol_key] = true_trace

          # Find best Proof for False
          false_trace = find_matching_derivation(
              rs.get("false_derivations", []), 
              context_set, 
              solution_vars
          )
          if false_trace:
              gt_q_to_false_derivation[sol_key] = false_trace
      
      # Data Recording
      num_rules = rs.get("depth", 0) 
      depth = rs.get("depth", 0)
      
      # Refine stats based on actual derivation found
      if gt_q_to_true_derivation:
          first_key = list(gt_q_to_true_derivation.keys())[0]
          num_rules = len(gt_q_to_true_derivation[first_key])
      elif gt_q_to_false_derivation:
          first_key = list(gt_q_to_false_derivation.keys())[0]
          num_rules = len(gt_q_to_false_derivation[first_key])
      
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
          search_space_vars,
          clean_gt_qs,
          gt_q_to_true_derivation,
          gt_q_to_false_derivation,
      ]

  # Save Output
  if len(data) > 0:
      prompt_indices = random.sample(range(len(data)), min(25, len(data)))
      prompts = data.iloc[prompt_indices]
      data_subsample = data.iloc[list(set(range(len(data))) - set(prompt_indices))]
      
      prompt_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_prompts.csv")
      data_path = os.path.join(arguments.sl_dir, "simplelogic_heldout_k_sufficient_data.csv")
      
      print(f"Writing {len(prompts)} prompts to {prompt_path}")
      with open(prompt_path, "w") as f:
        prompts.to_csv(f, index=False)
        
      print(f"Writing {len(data_subsample)} rows to {data_path}")
      with open(data_path, "w") as f:
        data_subsample.to_csv(f, index=False)
  else:
      print("No valid data rows generated.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--sl_dir",
      default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/archived",
      help="Directory containing the SimpleLogic data.",
  )
  parser.add_argument(
      "--max_problems_to_sample_per_ruleset",
      default=50,
      type=int,
      help="Maximum number of problems to sample per ruleset.",
  )
  main(parser.parse_args())