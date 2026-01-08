import ast
import collections
import itertools

def parse_clauses(rules_str):
    """
    Parses rule strings like "['a', 'not b', 'c']" into logical clauses.
    Returns: a list of sets, e.g., [{('a', True), ('b', False), ('c', True)}]
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
    NOTE: Unit propagation only infers facts when a clause has all but one literal false. This is valid because we only have Horn clauses in SimpleLogic.
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

    known_true = ast.literal_eval(data.get('known_facts', '[]'))
    known_false = ast.literal_eval(data.get('known_untrue_facts', '[]')) if 'known_untrue_facts' in data else []

    context = {k: True for k in known_true}
    for k in known_false:
        context[k] = False

    goal = data['goal']
    gt_qs_list = ast.literal_eval(data['gt_qs'])
    if isinstance(gt_qs_list[0], str):
        gt_qs_list = [[gt_q] for gt_q in gt_qs_list]  # Compatible with original questbench k=1 format

    # Check 1: Context underspecified (goal should not already be known)
    base_facts = solve_unit_prop(clauses, context)
    if base_facts != "CONTRADICTION" and goal in base_facts:
        print("❌ FAIL: Goal is already known from context.")
        if data["k"] not in counter['failed (goal inferred from context)'].keys():
            counter['failed (goal inferred from context)'][data["k"]] = 0
        counter['failed (goal inferred from context)'][data["k"]] += 1
        return False

    def build_table(qs):
        """Return (table, any_consistent). table maps answer tuples -> goal boolean."""
        table = {}
        any_consistent = False
        for answers in itertools.product([True, False], repeat=len(qs)):
            hyp_context = context.copy()
            hyp_context.update(zip(qs, answers))
            result = solve_unit_prop(clauses, hyp_context)
            if result == "CONTRADICTION":
                continue
            any_consistent = True
            if goal not in result:
                return None, True  # insufficient
            table[answers] = result[goal]
        return table, any_consistent

    def subset_is_sufficient(subset):
        table, any_consistent = build_table(subset)
        if table is None:
            return False  # insufficient (some consistent branch doesn't determine)
        return any_consistent  # sufficient iff there exists at least one consistent branch and all determine

    def essentiality_holds(table, qs):
        k = len(qs)
        for i in range(k):
            essential = False
            for others in itertools.product([True, False], repeat=k-1):
                t0 = []
                t1 = []
                j = 0
                for p in range(k):
                    if p == i:
                        t0.append(False)
                        t1.append(True)
                    else:
                        t0.append(others[j])
                        t1.append(others[j])
                        j += 1
                t0 = tuple(t0)
                t1 = tuple(t1)
                if t0 in table and t1 in table and table[t0] != table[t1]:
                    essential = True
                    break
            if not essential:
                return False
        return True

    for qs in gt_qs_list:
        # Check 2: Sufficiency (all consistent branches must determine goal)
        table, any_consistent = build_table(qs)
        if table is None:
            print(f"❌ FAIL: Some consistent branch for qs={qs} does not determine the goal.")
            if data["k"] not in counter['failed (insufficient branch)'].keys():
                counter['failed (insufficient branch)'][data["k"]] = 0
            counter['failed (insufficient branch)'][data["k"]] += 1
            return False
        if not any_consistent:
            print(f"❌ FAIL: All branches for qs={qs} are contradictions (degenerate problem).")
            if data["k"] not in counter['failed (insufficient branch)'].keys():
                counter['failed (insufficient branch)'][data["k"]] = 0
            counter['failed (insufficient branch)'][data["k"]] += 1
            return False

        # Check 3: Essentiality (each variable must be essential under consistent assignments)
        if len(qs) > 1 and not essentiality_holds(table, qs):
            print(f"❌ FAIL: qs={qs} is sufficient but not minimal (violates essentiality).")
            print(f"Ground truth qs: {qs}")
            print(f"Context: {context}")
            print(f"Rules: {clauses}")
            print(f"Goal: {goal}")
            print(f"Table: {table}")
            if data["k"] not in counter['failed (not minimal)'].keys():
                counter['failed (not minimal)'][data["k"]] = 0
            counter['failed (not minimal)'][data["k"]] += 1
            return False

        # Optional Check 4: Local minimality by subsets of size k-1
        for i in range(len(qs)):
            subset = qs[:i] + qs[i+1:]
            if subset and subset_is_sufficient(subset):
                print(f"❌ FAIL: Subset {subset} is sufficient (Not Minimal).")
                if data["k"] not in counter['failed (not minimal)'].keys():
                    counter['failed (not minimal)'][data["k"]] = 0
                counter['failed (not minimal)'][data["k"]] += 1
                return False

    print(f"✅ Row Verified (k={data['k'] if 'k' in data else 'N/A'})")
    if data["k"] not in counter['verified'].keys():
        counter['verified'][data["k"]] = 0
    counter['verified'][data["k"]] += 1
    return True


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    args = ArgumentParser()
    args.add_argument("--input_csv", type=str, default="/n/holylfs06/LABS/mzitnik_lab/Lab/yeh803/Reasoning/benchmark_data/questbench_data/Logic-Q/RP/RP/new_11_500k/simplelogic_heldout_k_sufficient_data_new.csv", help="Path to the CSV file with results to verify.")
    arguments = args.parse_args()

    counter = {'verified': {1:0}, 'failed (goal inferred from context)': {1:0}, 'failed (insufficient branch)': {1:0}, 'failed (not minimal)': {1:0}}
    data = pd.read_csv(arguments.input_csv)
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Verifying results"):
        print(f"Verifying row {idx}:")
        verify_row(row, counter)

    print("\nVerification Summary:")
    for key, value in counter.items():
        print(f"{key}: {value}")