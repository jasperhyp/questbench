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

"""Utilities for generating ambiguous questions by holding out 1 word."""

import copy
import itertools as it
import collections
import json

import tqdm

from SimpleLogic import derivation_new
from SimpleLogic import ruleset


def check_sufficiency(rule_tree, context, vars_to_check, target):
  """Checks if vars_to_check are sufficient to determine target given context.

  Args:
    rule_tree: RuleTree
    context: Set[str] of facts (e.g. {"a", "not b"})
    vars_to_check: Set[str] of variable names (e.g. {"c", "d"})
    target: str target variable name

  Returns:
    bool: True if for all consistent assignments of vars_to_check, the target's
      truth value is determined. False otherwise.
  """
  var_list = list(vars_to_check)
  # Iterate over all possible truth value assignments for vars_to_check
  for values in it.product([True, False], repeat=len(var_list)):
    assignment_facts = set()
    for i, val in enumerate(values):
      if val:
        assignment_facts.add(var_list[i])
      else:
        assignment_facts.add(ruleset.negate(var_list[i]))

    full_facts = context.union(assignment_facts)

    # Split into true/false for get_all_inferrable_facts
    true_facts = {f for f in full_facts if not f.startswith("not ")}
    false_facts = {f for f in full_facts if f.startswith("not ")}

    # Check for immediate contradiction in the input facts
    contradiction = False
    for f in true_facts:
      if ruleset.negate(f) in full_facts:
        contradiction = True
        break
    if contradiction:
      continue  # Vacuously true if the assignment is impossible

    inferred = get_all_inferrable_facts(rule_tree, true_facts, false_facts)

    # Check for consistency in inferred facts
    for f in inferred:
      if ruleset.negate(f) in inferred:
        contradiction = True
        break
    if contradiction:
      continue

    # If consistent, check if target is determined
    if target not in inferred and ruleset.negate(target) not in inferred:
      return False

  return True


def make_heldout_ruleset(rules_dict, max_k=4):
  """Hold out k words required for deriving the target word's truth value.

  Keeps only those derivations that are not already derived by the full set.
  Adds field to rules_dict called `heldout_set_to_q` and
  `heldout_set_to_subset_qs`
  `heldout_set_to_q` maps a heldout set to a dict of the form
  {1-sufficient set: sufficient word: {true_derivation, false_derivation}}
  where 1-sufficient set is missing 1 word that is required for deriving the
  target word's truth value, and sufficient word is a word that when known, can
  derive the target word's truth value. The true/false derivations are the
  derivations of the target word's truth value after adding the sufficient word
  to the heldout set.
  `heldout_set_to_subset_qs` maps a 1-sufficient set to a list of heldout sets
  that are subsets of the heldout set, but not the heldout set itself.

  Also generates k-sufficient sets for k=2 to max_k.
  
  Args:
    rules_dict: Dict[str, Any]
    max_k: int, max size of heldout set to search for (default 1, supports up to 3)
  """
  assert max_k in [1, 2, 3, 4], "k_max must be 1, 2, 3, or 4."
  
  print("Generating k=1")
  # UNCHANGED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  true_derivations = {}
  for derive in rules_dict['true_derivations']:
    if not isinstance(derive, derivation_new.ConjunctionRule):
      derive = derivation_new.ConjunctionRule.deserialize(derive)
    true_derivations[frozenset(derive.leaf_words.keys())] = derive
  false_derivations = {}
  for derive in rules_dict['false_derivations']:
    if not isinstance(derive, derivation_new.ConjunctionRule):
      derive = derivation_new.ConjunctionRule.deserialize(derive)
    false_derivations[frozenset(derive.leaf_words.keys())] = derive

  heldout_set_to_q_depth_expansions = {}
  # Pre-calculate RuleTree for inference checks
  # rule_tree = ruleset.RuleTree.deserialize(rules_dict["rules"])
  
  # get combined version (cross product)
  for true_derive, false_derive in tqdm.tqdm(
      it.product(true_derivations, false_derivations),
      total=len(true_derivations) * len(false_derivations),
  ):
    # find pairs that differ on exactly 1 var
    differ_word = None
    for k in true_derive:
      if ruleset.negate(k) in false_derive:
        if differ_word is None:
          differ_word = k
        elif differ_word != k:
          differ_word = None
          break
    if differ_word is None:
      continue

    heldout_set = true_derive.union(false_derive)
    heldout_set -= {differ_word, ruleset.negate(differ_word)}
    # NOTE: heldout_set means context (Assigned)

    # can already derive heldout set
    # 1. Check if Context ALREADY implies the answer (0-sufficient)
    skip_this = False
    if heldout_set in true_derivations or heldout_set in false_derivations:
      skip_this = True
    else:
      for true_derivation in true_derivations:
        if true_derivation <= heldout_set:
          skip_this = True
          break
      for false_derivation in false_derivations:
        if false_derivation <= heldout_set:
          skip_this = True
          break
    if skip_this:
      continue
    
    # 2. Check Strict K-Sufficiency (Ensure k-1 vars are NOT sufficient)
    if heldout_set not in heldout_set_to_q_depth_expansions:
      heldout_set_to_q_depth_expansions[heldout_set] = {}
    # add variable with direction that implies query word is true
    heldout_set_to_q_depth_expansions[heldout_set][differ_word] = {
        'true_derivation': true_derivations[true_derive].serialize(),
        'false_derivation': false_derivations[false_derive].serialize(),
    }

  rules_dict['heldout_set_to_q'] = {
      json.dumps(list(heldout_set)): heldout_set_to_q_depth_expansions[
          heldout_set
      ]
      for heldout_set in heldout_set_to_q_depth_expansions
  }

  # valid questions of subset --> also valid questions here,
  # but we want to avoid asking
  heldout_set_to_subset_qs = {}
  for heldout_set in heldout_set_to_q_depth_expansions:
    heldout_set_to_subset_qs[heldout_set] = set()
    for other_set in heldout_set_to_q_depth_expansions:
      if heldout_set == other_set:
        continue
      # check if other_set is a subset of heldout_Set  # NOTE: so that the heldout_set itself does not imply answer on its own
      if other_set < heldout_set:
        heldout_set_to_subset_qs[heldout_set] = heldout_set_to_subset_qs[
            heldout_set
        ].union(set(heldout_set_to_q_depth_expansions[other_set].keys()))
      # remove target_attr from set
      if rules_dict['query'] in heldout_set_to_subset_qs[heldout_set]:
        heldout_set_to_subset_qs[heldout_set].remove(rules_dict['query'])
  rules_dict['heldout_set_to_subset_qs'] = {
      json.dumps(list(heldout_set)): list(heldout_set_to_subset_qs[heldout_set])
      for heldout_set in heldout_set_to_subset_qs
  }
  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  
  # --- Extension for k-sufficient sets ---
  # Initialize with 1-sufficient sets found above
  k_sufficient_sets = {1: []}
  for heldout_set, val in heldout_set_to_q_depth_expansions.items():
    for word in val:
      var_name = word.split("not ")[1] if word.startswith("not ") else word
      k_sufficient_sets[1].append((heldout_set, {var_name}))

  rule_tree_obj = rules_dict['rules']
  if not isinstance(rule_tree_obj, ruleset.RuleTree):
    # Should typically be RuleTree if called within pipeline, but ensure safety
    rule_tree_obj = ruleset.RuleTree.deserialize(rules_dict['rules'])

  for k in range(2, max_k + 1):
    temp_k_sets = []
    seen_sigs = set()
    prev_items = k_sufficient_sets[k - 1]
        
    # Pre-group the previous sets by context to speed up the global minimality check
    # context (frozenset) -> list of s_sets (set)
    context_to_s_sets = collections.defaultdict(list)
    for context_prev, s_prev in prev_items:
        context_to_s_sets[context_prev].append(s_prev)
        
    # Pre-collect all lower level contexts for global minimality
    lower_k_contexts = set()
    for j in range(1, k):
        for context_prev, _ in k_sufficient_sets[j]:
            lower_k_contexts.add(context_prev)

    for context_prev, s_prev in tqdm.tqdm(prev_items, desc=f"Generating k={k}"):
      # Iterate through each fact in the previous context
      for fact in list(context_prev):
        var_name = fact.split("not ")[1] if fact.startswith("not ") else fact

        # Construct candidate next sets
        context_k = context_prev - {fact}
        s_k = s_prev | {var_name}

        # Avoid processing same (context, s_set) pair multiple times
        sig = (frozenset(context_k), frozenset(s_k))
        if sig in seen_sigs:
          continue
        seen_sigs.add(sig)

        # We construct the "flipped" version of the removed fact
        # to check sufficiency conditions
        flipped_fact = ruleset.negate(fact)

        # Condition 1: A_k + Known(S_{k-1}) + flipped_fact => Known(y)
        # s_prev corresponds to S_{k-1}
        if not check_sufficiency(
            rule_tree_obj,
            context_k | {flipped_fact},
            s_prev,
            rules_dict['query'],
        ):
          continue

        # Condition 2 & 4 combined (Global Minimality & Underspecification):
        # We need to ensure that NO set of size k-1 determines y under the new reduced context A_k.
        is_lower_k_sufficient = False
        
        # Check A: Subset check
        # If any subset of context_k is a known minimal context for size k-1 (or less),
        # then context_k is solvable by size k-1 (monotonicity).
        for c_existing in context_to_s_sets:
            if c_existing <= context_k:
                is_lower_k_sufficient = True
                break
        
        if is_lower_k_sufficient:
            continue

        # Check B: Parent Context Check
        # Check all sufficient sets of the parent context (size k-1)
        # A sufficient set that solves context_k should also solve context_prev
        for s_other in context_to_s_sets[context_prev]:
            # If s_other solves the problem under context_k, then A_k 
            # is solvable with k-1 vars, so S_k (size k) is not minimal.
            if check_sufficiency(
                rule_tree_obj, context_k, s_other, rules_dict['query']
            ):
                is_lower_k_sufficient = True
                break
        
        if is_lower_k_sufficient:
            continue

        # # Condition 3: Local minimality -- UNNECESSARY AFTER HAVING GLOBAL MINIMALITY
        # # For all Z subset S_{k-1} (size k-2):
        # # A_k + Known(Z) + flipped_fact => not Known(y)
        # minimality_fail = False
        # for z in s_prev:
        #   z_subset = s_prev - {z}
        #   if check_sufficiency(
        #       rule_tree_obj,
        #       context_k | {flipped_fact},
        #       z_subset,
        #       rules_dict['query'],
        #   ):
        #     minimality_fail = True
        #     break
        # if minimality_fail:
        #   continue
        
        # NOTE: This set is to ensure sets at the same level won't be used for this check.
        temp_k_sets.append((context_k, s_k))

    # k_sufficient_sets[k].append((context_k, s_k))
    k_sufficient_sets[k] = temp_k_sets

  # Store k-sufficient sets in the rules_dict
  rules_dict['heldout_k_sets'] = {}
  for k in range(1, max_k + 1):
    res_map = {}
    for context, s_set in k_sufficient_sets[k]:
      c_str = json.dumps(sorted(list(context)))
      if c_str not in res_map:
        res_map[c_str] = []
      res_map[c_str].append(sorted(list(s_set)))
    rules_dict['heldout_k_sets'][str(k)] = res_map


def get_all_inferrable_facts(rule_tree, true_facts, false_facts):
  """Get all facts that can be inferred from the given true and false facts.

  Args:
    rule_tree: RuleTree
    true_facts: Set[str]
    false_facts: Set[str]

  Returns:
    Set[str]
  """
  # ASSUMES FALSE_FACTS ALREADY IN FORM "not _"
  rules_to_consider = copy.deepcopy(rule_tree.nodes)

  all_facts = set(true_facts).union(set(false_facts))

  curr_facts = all_facts
  while curr_facts:
    new_facts = set()
    for fact in curr_facts:
      if ruleset.negate(fact) not in rules_to_consider:
        continue
      for rule in rules_to_consider[ruleset.negate(fact)].rules:
        # find any facts that must be true, due to negate(fact) being false
        remaining_terms = set(rule)
        rule_true = False
        for term in rule:
          if term in all_facts:
            # this rule is true by this term
            rule_true = True
            break
          if ruleset.negate(term) in all_facts:
            # other terms must be true
            remaining_terms.remove(term)
        if rule_true:
          continue
        if len(remaining_terms) == 1:
          new_facts.add(remaining_terms.pop())
    curr_facts = new_facts
    all_facts = all_facts.union(curr_facts)
  return all_facts
