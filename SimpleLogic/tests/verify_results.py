import ast
import collections
import itertools

def parse_clauses(rules_str):
    """
    Parses rule strings like "['a', 'not b', 'c']" into logical clauses.
    Returns a list of sets, e.g., [{('a', True), ('b', False), ('c', True)}]
    """
    if isinstance(rules_str, str):
        raw_rules = ast.literal_eval(rules_str)
    else:
        raw_rules = rules_str

    clauses = []
    for rule in raw_rules:
        clause = set()
        for lit in rule:
            if lit.startswith("not "):
                clause.add((lit[4:], False))
            else:
                clause.add((lit, True))
        clauses.append(clause)
    return clauses

def solve_unit_prop(clauses, context):
    """
    Performs Unit Propagation to derive all implied facts.
    Returns: dict of derived facts, or "CONTRADICTION".
    """
    assignment = context.copy()
    
    while True:
        changed = False
        for clause in clauses:
            # Check if clause is satisfied or if we can infer a new fact
            satisfied = False
            unknown_lits = []
            false_lits_count = 0
            
            for var, val in clause:
                if var in assignment:
                    if assignment[var] == val:
                        satisfied = True
                        break
                    else:
                        false_lits_count += 1
                else:
                    unknown_lits.append((var, val))
            
            if satisfied:
                continue
            
            if false_lits_count == len(clause):
                return "CONTRADICTION"
            
            # If only 1 unknown literal remains and all others are false, infer it
            if len(unknown_lits) == 1:
                var, val = unknown_lits[0]
                if var not in assignment:
                    assignment[var] = val
                    changed = True
                elif assignment[var] != val:
                    return "CONTRADICTION"
        
        if not changed:
            break
            
    return assignment

def verify_row(data, counter):
    clauses = parse_clauses(data['rules'])
    known = ast.literal_eval(data['known_facts'])
    context = {k: True for k in known}
    goal = data['goal']
    gt_qs_list = ast.literal_eval(data['gt_qs'])
    if isinstance(gt_qs_list[0], str):
        gt_qs_list = [[gt_q] for gt_q in gt_qs_list]  # Compatible with original questbench k=1 format
    
    # Check 1: Context Underspecified
    base_facts = solve_unit_prop(clauses, context)
    if goal in base_facts:
        print("❌ FAIL: Goal is already known from context.")
        counter['failed (goal inferred from context)'] += 1
        return False

    for qs in gt_qs_list:
        # Check 2: Sufficiency (All 2^k branches must solve goal)
        for answers in itertools.product([True, False], repeat=len(qs)):
            hyp_context = context.copy()
            hyp_context.update(zip(qs, answers))
            result = solve_unit_prop(clauses, hyp_context)
            
            if result != "CONTRADICTION" and goal not in result:
                print(f"❌ FAIL: Branch {dict(zip(qs, answers))} is insufficient.")
                counter['failed (insufficient branch)'] += 1
                return False

        # Check 3: (Local) Minimality (Subsets must be insufficient)
        for i in range(len(qs)):
            subset = qs[:i] + qs[i+1:]
            subset_sufficient = True
            for answers in itertools.product([True, False], repeat=len(subset)):
                hyp_context = context.copy()
                hyp_context.update(zip(subset, answers))
                result = solve_unit_prop(clauses, hyp_context)
                if result != "CONTRADICTION" and goal not in result:
                    subset_sufficient = False
                    break
            
            if subset_sufficient:
                print(f"❌ FAIL: Subset {subset} is sufficient (Not Minimal).")
                counter['failed (not minimal)'] += 1
                return False

    print(f"✅ Row Verified (k={data['k'] if 'k' in data else 'N/A'})")
    counter['verified'] += 1
    return True

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    args = ArgumentParser()
    args.add_argument("--input_csv", type=str, default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k/simplelogic_heldout_k_sufficient_data_new.csv", help="Path to the CSV file with results to verify.")
    arguments = args.parse_args()

    counter = {'verified': 0, 'failed (goal inferred from context)': 0, 'failed (insufficient branch)': 0, 'failed (not minimal)': 0}
    data = pd.read_csv(arguments.input_csv)
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Verifying results"):
        print(f"Verifying row {idx}:")
        verify_row(row, counter)

    print("\nVerification Summary:")
    for key, value in counter.items():
        print(f"{key}: {value}")