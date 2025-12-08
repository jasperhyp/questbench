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
import json

import tqdm

from SimpleLogic import derivation
from SimpleLogic import ruleset


def get_vars_from_literals(literals):
  """Extract variable names from a set of literals (e.g. {'not a'} -> {'a'})."""
  vars_set = set()
  for l in literals:
    if l.startswith("not "):
      vars_set.add(l[4:])
    else:
      vars_set.add(l)
  return vars_set


def verify_k_sufficiency(
    k_vars,
    base_assignment,
    true_derivations,
    false_derivations,
):
  """Verifies if k_vars is a k-sufficient set given a base assignment.

  Implements the 'Mask and Verify' logic:
  1. Generate all 2^k assignments for k_vars.
  2. For each assignment, check if (base + assignment) implies True or False.
  3. It must imply True for some assignments and False for others (mixed results),
     but every assignment must yield a definitive result (no ambiguity).

  Args:
    k_vars: Set[str] variables to test.
    base_assignment: Set[str] known literals (context).
    true_derivations: List[Set[str]] all full assignments implying True.
    false_derivations: List[Set[str]] all full assignments implying False.

  Returns:
    bool: True if k_vars is k-sufficient.
  """
  # Sort vars to ensure deterministic iteration order for binary string generation
  sorted_vars = sorted(list(k_vars))
  
  has_true_outcome = False
  has_false_outcome = False

  # Iterate 2^k assignments
  for i in range(1 << len(sorted_vars)):
    current_assignment = set(base_assignment)
    
    # Construct specific assignment for k_vars
    for idx, var in enumerate(sorted_vars):
      if (i >> idx) & 1:
        current_assignment.add(var)
      else:
        current_assignment.add(f"not {var}")
    
    # Check if this fully specified state implies True or False
    implies_true = False
    for t_deriv in true_derivations:
      if t_deriv.issubset(current_assignment):
        implies_true = True
        break
    
    implies_false = False
    for f_deriv in false_derivations:
      if f_deriv.issubset(current_assignment):
        implies_false = True
        break
    
    # Constraint 1: Completeness - must imply something
    if not implies_true and not implies_false:
      return False
    
    # Constraint 2: Consistency - cannot imply both (should be handled by CSP generation, but good sanity check)
    if implies_true and implies_false:
      return False

    if implies_true:
      has_true_outcome = True
    if implies_false:
      has_false_outcome = True

  # Constraint 3: Sensitivity - must be able to reach both True and False outcomes
  # (Otherwise, the base_assignment was already sufficient, or these vars don't matter)
  if not (has_true_outcome and has_false_outcome):
    return False

  return True

def make_heldout_ruleset(rules_dict, k_max=1):
  """Hold out k words required for deriving the target word's truth value.

  Uses 'Parity Hunter' heuristics to efficiently find 1, 2, and 3-sufficient sets.
  
  Args:
    rules_dict: Dict[str, Any]
    k_max: int, max size of heldout set to search for (default 1, supports up to 3)

  Keeps only those derivations that are not already derived by the full set.
  Adds field to rules_dict called `heldout_set_to_q` and
  `heldout_set_to_subset_qs`
  `heldout_set_to_q` maps a heldout set to a dict of the form
  {k-sufficient set: sufficient words (& string): {true_derivation, false_derivation}}
  where k-sufficient set is missing k word that is required for deriving the
  target word's truth value, and sufficient word is a word that when known, can
  derive the target word's truth value. The true/false derivations are the
  derivations of the target word's truth value after adding the sufficient word
  to the heldout set.
  `heldout_set_to_subset_qs` maps a 1-sufficient set to a list of heldout sets
  that are subsets of the heldout set, but not the heldout set itself.
  """
  assert k_max in [1, 2, 3], "k_max must be 1, 2, or 3."
  
  true_derivations = {}
  for derive in rules_dict['true_derivations']:
    if not isinstance(derive, derivation.ConjunctionRule):
      derive = derivation.ConjunctionRule.deserialize(derive)
    true_derivations[frozenset(derive.leaf_words.keys())] = derive
  false_derivations = {}
  for derive in rules_dict['false_derivations']:
    if not isinstance(derive, derivation.ConjunctionRule):
      derive = derivation.ConjunctionRule.deserialize(derive)
    false_derivations[frozenset(derive.leaf_words.keys())] = derive

  heldout_set_to_q_depth_expansions = {}
  # Pre-calculate RuleTree for inference checks
  # rule_tree = ruleset.RuleTree.deserialize(rules_dict["rules"])
  
  # get combined version (cross product)
  for true_derive, false_derive in tqdm.tqdm(
      it.product(true_derivations, false_derivations),
      total=len(true_derivations) * len(false_derivations),
  ):
    # FIXME
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # find pairs that differ on exactly 1 var
    # differ_word = None
    # for k in true_derive:
    #   if ruleset.negate(k) in false_derive:
    #     if differ_word is None:
    #       differ_word = k
    #     elif differ_word != k:
    #       differ_word = None
    #       break
    # if differ_word is None:
    #   continue

    # heldout_set = true_derive.union(false_derive)
    # heldout_set -= {differ_word, ruleset.negate(differ_word)}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # # Identify all conflicting variables
    # differing_words = set()
    # for k in true_derive:
    #   if ruleset.negate(k) in false_derive:
    #     base_var = k if not k.startswith("not ") else k[4:]
    #     differing_words.add(base_var)
            
    # # Only proceed if we have exactly k differences
    # if len(differing_words) != k_sufficient:
    #   continue
    
    # # For k=1 we still want a single literal key (with polarity),
    # # matching the original behavior (differ_word).
    # differ_word = None
    # if k_sufficient == 1:
    #   base = next(iter(differing_words))
    #   # In the original code differ_word is the literal in true_derive.
    #   differ_word = base if base in true_derive else ruleset.negate(base)
      
    # # Construct the base heldout set (Context)
    # # Remove all conflicting vars (both positive and negative forms)
    # heldout_set = true_derive.union(false_derive)
    # for word in differing_words:
    #     heldout_set -= {word, ruleset.negate(word)}
    
    
    
    
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # can already derive heldout set
    # 1. Check if Context ALREADY implies the answer (0-sufficient)
    # UNCHANGED
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
    
    # FIXME
    # 2. Check Strict K-Sufficiency (Ensure k-1 vars are NOT sufficient)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # if heldout_set not in heldout_set_to_q_depth_expansions:
    #   heldout_set_to_q_depth_expansions[heldout_set] = {}
    # # add variable with direction that implies query word is true
    # heldout_set_to_q_depth_expansions[heldout_set][differ_word] = {
    #     'true_derivation': true_derivations[true_derive].serialize(),
    #     'false_derivation': false_derivations[false_derive].serialize(),
    # }
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # ---------------------------------------------------------------------
    # Strict k-sufficiency for k>1:
    #    No proper subset of the k missing variables, under any
    #    polarity assignment, should be sufficient to derive the query.
    #
    #    For k=1 there is no non-empty proper subset, so we keep
    #    the original behavior and skip this check.
    # ---------------------------------------------------------------------
    # if k_sufficient > 1:
    #   is_strictly_k = True
    #   base_list = sorted(differing_words)

    #   # r = subset size, 1..k-1
    #   for r in range(1, k_sufficient):
    #     for subset in it.combinations(base_list, r):
    #       # Enumerate all 2^r polarity assignments for this subset.
    #       for polarity_bits in it.product([0, 1], repeat=r):
    #         subset_lits = set()
    #         for var, bit in zip(subset, polarity_bits):
    #           lit = var if bit == 1 else ruleset.negate(var)
    #           subset_lits.add(lit)

    #         test_context = heldout_set.union(subset_lits)

    #         # Does this partial information derive the query (true or false)?
    #         solves_true = any(td <= test_context for td in true_derivations)
    #         solves_false = any(fd <= test_context for fd in false_derivations)

    #         if solves_true or solves_false:
    #           is_strictly_k = False
    #           break

    #       if not is_strictly_k:
    #         break
    #     if not is_strictly_k:
    #       break

    #   if not is_strictly_k:
    #     continue
      
    # if heldout_set not in heldout_set_to_q_depth_expansions:
    #   heldout_set_to_q_depth_expansions[heldout_set] = {}
    
    # # Store the set of variables as a combined string key
    # # e.g., "happy & smart"
    # if k_sufficient == 1:
    #   q_key = differ_word
    # else:
    #   # A simple combined name for the multi-variable question.
    #   q_key = " & ".join(sorted(differing_words))

    # heldout_set_to_q_depth_expansions[heldout_set][q_key] = {
    #     'true_derivation': true_derivations[true_derive].serialize(),
    #     'false_derivation': false_derivations[false_derive].serialize(),
    # }
    
    
    
    
    
    
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
