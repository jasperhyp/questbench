"""Profile the derivation generation to identify bottlenecks."""

import cProfile
import pstats
import io
import time
from functools import wraps

from SimpleLogic import derivation
from SimpleLogic import ruleset


def time_section(name):
    """Context manager to time a section of code."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"Starting: {name}")
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            print(f"Finished: {name} in {self.end - self.start:.2f}s")
    
    return Timer()


def profile_backderive_detailed(rules_dict):
    """Add detailed timing to backderive_nextlayer_rules."""
    import itertools as it
    
    if not isinstance(rules_dict["rules"], ruleset.RuleTree):
        rules_dict["rules"] = ruleset.RuleTree.deserialize(rules_dict["rules"])
    
    rule_tree = rules_dict["rules"]
    query = rules_dict["query"]
    
    if query not in rule_tree.nodes:
        print("Query not in rule tree")
        return
    
    prev_layer_rules = {
        derivation.ConjunctionRule({query: 0}, {}, [], query, 0)
    }
    all_query_rules = set()
    max_depth = len(rule_tree.nodes)
    
    print(f"Max depth: {max_depth}")
    print(f"Number of nodes: {len(rule_tree.nodes)}")
    
    # Count rules per node
    total_rules = sum(len(node.rules) for node in rule_tree.nodes.values())
    print(f"Total rules across all nodes: {total_rules}")
    
    for depth in range(max_depth):
        print(f"\n{'='*50}")
        print(f"DEPTH {depth}")
        print(f"{'='*50}")
        print(f"prev_layer_rules: {len(prev_layer_rules)}")
        print(f"all_query_rules: {len(all_query_rules)}")
        
        if not prev_layer_rules:
            print("No more rules to expand")
            break
        
        # Time expansion phase
        curr_layer_rules = []
        expansion_time = 0
        product_sizes = []
        
        with time_section("Expansion phase"):
            for prev_layer_rule in prev_layer_rules:
                perword_expansion_rules = []
                prev_layer_words = []
                tree_depth = max(prev_layer_rule.leaf_words.values())
                
                t0 = time.perf_counter()
                for prev_layer_word in prev_layer_rule:
                    perword_expansion_rules.append(
                        [{prev_layer_word: prev_layer_rule.leaf_words[prev_layer_word]}]
                    )
                    if prev_layer_rule.leaf_words[prev_layer_word] < tree_depth:
                        continue
                    
                    for word_expansion in rule_tree.nodes[prev_layer_word].rules:
                        word_expansion = {
                            ruleset.negate(word): tree_depth + 1
                            for word in word_expansion
                            if word != prev_layer_word
                        }
                        if prev_layer_rule.list_has_ancestor(set(word_expansion.keys())):
                            continue
                        if prev_layer_rule.contradicts(set(word_expansion.keys())):
                            continue
                        perword_expansion_rules[-1].append(word_expansion)
                    prev_layer_words.append(prev_layer_word)
                
                # Calculate product size BEFORE iterating
                product_size = 1
                for exp_list in perword_expansion_rules:
                    product_size *= len(exp_list)
                product_sizes.append(product_size)
                
                # Time the product iteration
                t1 = time.perf_counter()
                count = 0
                for expansion in it.product(*perword_expansion_rules):
                    count += 1
                    new_leaves = derivation.union(expansion)
                    if set(new_leaves.keys()) >= prev_layer_rule:
                        continue
                    new_ancestors = {
                        **prev_layer_rule.ancestor_words,
                        **prev_layer_rule.leaf_words,
                    }
                    for k in derivation.union(expansion):
                        if k in new_ancestors:
                            del new_ancestors[k]
                    new_derivations = prev_layer_rule.derivation + [
                        (tuple(expansion[w].keys()), word)
                        for w, word in enumerate(prev_layer_words)
                        if not (len(expansion[w]) == 1 and word in expansion[w])
                    ]
                    curr_layer_rules.append(
                        derivation.ConjunctionRule(
                            new_leaves,
                            new_ancestors,
                            new_derivations,
                            prev_layer_rule.root_word,
                            prev_layer_rule.layer + 1,
                        )
                    )
                t2 = time.perf_counter()
                expansion_time += (t2 - t0)
        
        print(f"Product sizes: min={min(product_sizes)}, max={max(product_sizes)}, "
              f"sum={sum(product_sizes)}, mean={sum(product_sizes)/len(product_sizes):.1f}")
        print(f"curr_layer_rules after expansion: {len(curr_layer_rules)}")
        
        # Time pruning phase
        with time_section("Pruning ancestor rules"):
            ancestor_query_rules_pruned = set()
            comparisons = 0
            for rule in all_query_rules:
                has_subset = False
                for rule2 in curr_layer_rules:
                    comparisons += 1
                    if rule == rule2:
                        continue
                    if rule2 < rule:
                        has_subset = True
                        break
                if not has_subset:
                    ancestor_query_rules_pruned.add(rule)
            print(f"Comparisons: {comparisons}")
        
        with time_section("Pruning current layer rules"):
            curr_layer_rules_pruned = set()
            comparisons = 0
            all_to_check = curr_layer_rules + list(all_query_rules)
            for r, rule in enumerate(curr_layer_rules):
                has_subset = False
                for rule2 in all_to_check:
                    comparisons += 1
                    if rule == rule2:
                        continue
                    if rule2 < rule:
                        has_subset = True
                        break
                if not has_subset:
                    curr_layer_rules_pruned.add(rule)
            print(f"Comparisons: {comparisons}")
            print(f"Pruned from {len(curr_layer_rules)} to {len(curr_layer_rules_pruned)}")
        
        all_query_rules = ancestor_query_rules_pruned.union(curr_layer_rules_pruned)
        prev_layer_rules = curr_layer_rules_pruned
    
    print(f"\n{'='*50}")
    print(f"FINAL: {len(all_query_rules)} derivations")
    return all_query_rules


def run_cprofile(rules_dict):
    """Run cProfile on get_derivations."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    derivation.get_derivations(rules_dict)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl_dir", type=str, 
                        default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP")
    parser.add_argument("--idx", type=int, default=0, help="Index of rules_dict to profile")
    parser.add_argument("--mode", type=str, choices=["detailed", "cprofile", "both"], default="both")
    args = parser.parse_args()
    
    # Load one rules_dict
    rules_dicts, _ = ruleset.load_data(args.sl_dir)
    
    if args.idx >= len(rules_dicts):
        print(f"Index {args.idx} out of range (max {len(rules_dicts)-1})")
        exit(1)
    
    rules_dict = rules_dicts[args.idx]
    print(f"Profiling rules_dict {args.idx}")
    print(f"Query: {rules_dict['query']}")
    
    if args.mode in ["detailed", "both"]:
        print("\n" + "="*60)
        print("DETAILED PROFILING")
        print("="*60)
        profile_backderive_detailed(rules_dict.copy())
    
    if args.mode in ["cprofile", "both"]:
        print("\n" + "="*60)
        print("CPROFILE")
        print("="*60)
        run_cprofile(rules_dict.copy())
        