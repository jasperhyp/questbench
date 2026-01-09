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

"""Multi-turn problem-solving evaluator for Logic-Q-multi."""

import argparse
import ast
import asyncio
import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import tqdm

from model_utils import (
    cached_generate,
    load_cache_file,
    LocalModelConfig,
    get_local_model_config,
    is_local_model,
    model_call_wrapper,
)


# ============ Data Classes ============

@dataclass
class World:
    """A world is a specific assignment of values to queried variables."""
    assignments: Dict[str, bool]  # var -> True/False
    target_value: str  # "goal" or "not goal"


@dataclass
class Sample:
    """A single problem instance."""
    sample_id: int
    rules: List[List[str]]  # CNF clauses
    rules_nl: str  # Natural language rules
    goal: str
    known_facts: List[str]
    known_untrue_facts: List[str]
    cannot_ask: Set[str]
    all_valid_qs: List[str]
    gt_min_sets: List[List[str]]  # Minimal sufficient sets
    worlds: List[World]
    k: int  # Minimal sufficient size


@dataclass
class TurnLog:
    """Log of a single turn."""
    turn: int
    questions: List[str]
    oracle_answer: str
    model_response: str


@dataclass 
class EpisodeResult:
    """Result of a single episode (sample + world)."""
    sample_id: int
    world: World
    final_answer: Optional[str] = None
    correct: bool = False
    answered: bool = False
    turns_used: int = 0
    questions_asked: List[str] = field(default_factory=list)
    turn_logs: List[TurnLog] = field(default_factory=list)
    final_response: str = ""
    budget_violated: bool = False


@dataclass
class Action:
    """Parsed action from model response."""
    type: str  # "QUESTION", "ANSWER", "RETRY"
    questions: List[str] = field(default_factory=list)
    value: Optional[str] = None


# ============ Oracle (Unit Propagation) ============

def parse_clauses(rules: List[List[str]]) -> List[Set[Tuple[str, bool]]]:
    """
    Parse rules into clauses for unit propagation.
    rules: list[list[str]] where each inner list is a CNF clause like
      ['c', 'not a', 'not b'] meaning (c ∨ ¬a ∨ ¬b)
    Returns: list[set[(var:str, val:bool)]]
    """
    clauses = []
    for rule in rules:
        clause = set()
        for lit in rule:
            if lit.startswith("not "):
                clause.add((lit[4:], False))
            else:
                clause.add((lit, True))
        clauses.append(clause)
    return clauses


def solve_unit_prop(clauses: List[Set[Tuple[str, bool]]], 
                    context: Dict[str, bool]) -> Dict[str, bool] | str:
    """
    Unit propagation to infer all derivable facts.
    Returns: dict[var -> bool] (closure) OR "CONTRADICTION"
    """
    assignment = dict(context)

    while True:
        changed = False
        for clause in clauses:
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

            # Empty clause under current partial assignment => contradiction
            if false_lits_count == len(clause):
                return "CONTRADICTION"

            # Unit clause => infer
            if len(unknown_lits) == 1 and false_lits_count == len(clause) - 1:
                var, val = unknown_lits[0]
                if var not in assignment:
                    assignment[var] = val
                    changed = True
                elif assignment[var] != val:
                    return "CONTRADICTION"

        if not changed:
            break

    return assignment


def oracle_answer(clauses: List[Set[Tuple[str, bool]]],
                  base_context: Dict[str, bool],
                  world: World,
                  questions: List[str]) -> str:
    """
    Answer questions based on world + unit propagation inference.
    Returns natural language answer string.
    """
    # Combine base context with world assignments
    full_context = dict(base_context)
    full_context.update(world.assignments)
    
    # Run unit propagation
    inferred = solve_unit_prop(clauses, full_context)
    if inferred == "CONTRADICTION":
        return "There is a contradiction in the given information."
    
    # Build answers
    answers = []
    for q in questions:
        # Normalize question (remove "not " prefix if present)
        var = q.replace("not ", "") if q.startswith("not ") else q
        
        if var in inferred:
            if inferred[var]:
                answers.append(f"Yes, Alice is {var}.")
            else:
                answers.append(f"No, Alice is not {var}.")
        else:
            answers.append(f"Not sure whether Alice is {var}.")
    
    return " ".join(answers)


# ============ Data Loading ============

def parse_rules_to_nl(rules: List[List[str]]) -> str:
    """Parse rules into natural language format."""
    rules_nl = []
    for rule in rules:
        negated_words = [
            word.split("not ")[-1] for word in rule if word.startswith("not ")
        ]
        positive_words = [word for word in rule if not word.startswith("not ")]
        if len(positive_words) != 1:
            continue
        premises = " and ".join(negated_words)
        conclusion_word = positive_words[0]
        rules_nl.append(f"If Alice is {premises}, then Alice is {conclusion_word}.")
    return "\n".join(sorted(rules_nl))


def parse_world_key(key_json: str) -> Dict[str, bool]:
    """Parse world key like '["zealous", "not stormy"]' to {var: bool}."""
    facts = json.loads(key_json)
    assignments = {}
    for f in facts:
        if f.startswith("not "):
            assignments[f[4:]] = False
        else:
            assignments[f] = True
    return assignments


def load_data(csv_path: str) -> List[Sample]:
    """Load data from CSV and parse into Sample objects."""
    df = pd.read_csv(csv_path)
    
    samples = []
    for idx, row in df.iterrows():
        # Parse string representations to Python objects
        rules = ast.literal_eval(row["rules"])
        known_facts = ast.literal_eval(row["known_facts"])
        known_untrue_facts = ast.literal_eval(row["known_untrue_facts"])
        cannot_ask = set(ast.literal_eval(row["cannot_ask_facts"]))
        all_valid_qs = ast.literal_eval(row["all_valid_qs"])
        gt_qs = ast.literal_eval(row["gt_qs"])
        
        # Parse worlds from gt_q_to_derivations_min_rules
        derivations = ast.literal_eval(row["gt_q_to_derivations_min_rules"])
        worlds = []
        # derivations is a list of dicts, one per gt_q set
        # Each dict maps assignment_key -> {"target_value": ..., "derivation": ...}
        for deriv_dict in derivations:
            for key_json, value in deriv_dict.items():
                assignments = parse_world_key(key_json)
                target_value = value["target_value"]
                worlds.append(World(assignments=assignments, target_value=target_value))
        
        sample = Sample(
            sample_id=idx,
            rules=rules,
            rules_nl=parse_rules_to_nl(rules),
            goal=row["goal"],
            known_facts=known_facts,
            known_untrue_facts=known_untrue_facts,
            cannot_ask=cannot_ask,
            all_valid_qs=all_valid_qs,
            gt_min_sets=gt_qs,
            worlds=worlds,
            k=int(row["k"]),
        )
        samples.append(sample)
    
    return samples


# ============ Prompting ============

SYSTEM_PROMPT_TEMPLATE = """Suppose you know the following rules about Alice:
{rules_nl}

You need to determine whether Alice is {goal}.

You may ask up to {budget} questions about Alice's attributes. Format questions as:
"Question: Is Alice [attribute]?"

When ready to answer, respond with:
"Answer: {goal}" or "Answer: not {goal}"

Do not output any other format."""

UNCERTAINTY_EMPHASIS = """

IMPORTANT: The initial facts provided are INSUFFICIENT to determine the answer. You MUST ask questions to gather the missing information before you can answer correctly. Do not guess - ask questions first."""


def build_initial_prompt(sample: Sample, budget: int, emphasize_uncertainty: bool = True) -> List[Dict[str, str]]:
    """Build initial prompt for the model."""
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        rules_nl=sample.rules_nl,
        goal=sample.goal,
        budget=budget,
    )
    
    # Add uncertainty emphasis if enabled
    if emphasize_uncertainty:
        system_content += UNCERTAINTY_EMPHASIS
    
    # Build known facts text
    facts = []
    for f in sample.known_facts:
        facts.append(f"Alice is {f}.")
    for f in sample.known_untrue_facts:
        facts.append(f"Alice is not {f}.")
    
    user_content = "\n".join(facts) if facts else "No facts are currently known about Alice."
    user_content += f"\n\nIs Alice {sample.goal}?"
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


FORCE_ANSWER_PROMPT = "\n\nYou have used all your questions. You must now provide your final answer in the format: Answer: [your answer]"

RETRY_PROMPT = """Could not parse your response. Please respond with exactly one of:
- "Question: Is Alice [attribute]?" to ask a question
- "Answer: [goal]" or "Answer: not [goal]" to provide your final answer"""


# ============ Action Parsing ============

def extract_non_thinking(response: str) -> str:
    """Strip thinking trace from response (for local models)."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response


def parse_action(response: str, goal: str) -> Action:
    """Parse model response into an action."""
    # Strip thinking trace
    clean_response = extract_non_thinking(response)
    
    # Check for Answer
    answer_match = re.search(r"Answer:\s*(.+?)(?:\.|$)", clean_response, re.IGNORECASE)
    if answer_match:
        answer_value = answer_match.group(1).strip()
        return Action(type="ANSWER", value=answer_value)
    
    # Check for Question(s)
    question_matches = re.findall(r"Is Alice (\w+(?:-\w+)?)\?", clean_response, re.IGNORECASE)
    if question_matches:
        return Action(type="QUESTION", questions=[q.lower() for q in question_matches])
    
    # Alternative question pattern: "Question: [attr]"
    alt_matches = re.findall(r"Question:\s*(?:Is Alice\s+)?(\w+(?:-\w+)?)", clean_response, re.IGNORECASE)
    if alt_matches:
        return Action(type="QUESTION", questions=[q.lower() for q in alt_matches])
    
    return Action(type="RETRY")


def check_answer_correct(answer: Optional[str], target_value: str, goal: str) -> bool:
    """Check if the model's answer matches the target."""
    if answer is None:
        return False
    
    answer_lower = answer.lower().strip()
    target_lower = target_value.lower().strip()
    
    # Normalize both to canonical form
    def normalize(s):
        s = s.strip()
        # Handle variations like "not bored" vs "bored" 
        return s
    
    answer_norm = normalize(answer_lower)
    target_norm = normalize(target_lower)
    
    # Direct match
    if answer_norm == target_norm:
        return True
    
    # Check if both refer to same truth value about the goal
    # e.g., answer="bored" target="bored" -> True
    # e.g., answer="not bored" target="not bored" -> True  
    # e.g., answer="bored" target="not bored" -> False
    # e.g., answer="not bored" target="bored" -> False
    
    goal_lower = goal.lower().strip()
    
    # Determine what the answer claims about the goal
    answer_says_goal_true = (answer_norm == goal_lower)
    answer_says_goal_false = (answer_norm == f"not {goal_lower}" or 
                               answer_norm == f"not {goal_lower}".replace("-", " "))
    
    # Determine what the target says about the goal
    target_says_goal_true = (target_norm == goal_lower)
    target_says_goal_false = (target_norm == f"not {goal_lower}" or
                               target_norm == f"not {goal_lower}".replace("-", " "))
    
    # Match if both say the same thing
    if answer_says_goal_true and target_says_goal_true:
        return True
    if answer_says_goal_false and target_says_goal_false:
        return True
    
    return False
    

# ============ Episode Rollout ============

def run_episode(
    model_name: str,
    sample: Sample,
    world: World,
    budget: int,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    max_retries: int = 3,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
) -> EpisodeResult:
    """Run a single multi-turn episode."""
    
    if verbose:
        print(f"\n--- Episode: sample={sample.sample_id}, world={world.assignments}, target={world.target_value} ---")
    
    # Build initial prompt
    messages = build_initial_prompt(sample, budget, emphasize_uncertainty=emphasize_uncertainty)
    clauses = parse_clauses(sample.rules)
    
    # Build base context from known facts
    base_context = {}
    for f in sample.known_facts:
        base_context[f] = True
    for f in sample.known_untrue_facts:
        base_context[f] = False
    
    result = EpisodeResult(
        sample_id=sample.sample_id,
        world=world,
    )
    
    retry_count = 0
    turn = 0
    
    while turn <= budget:
        if verbose:
            print(f"  Turn {turn}/{budget}...")
        
        # Force answer on last turn
        current_messages = list(messages)
        if turn == budget:
            current_messages[-1] = dict(current_messages[-1])
            current_messages[-1]["content"] += FORCE_ANSWER_PROMPT
        
        # Generate response
        if verbose:
            print(f"    Generating response...")
        responses, thinking_tokens, cots, costs = cached_generate(
            [current_messages],
            model_name,
            cache=cache,
            cache_file=cache_file,
            generation_config=generation_config,
        )
        response = responses[0]
        if verbose:
            print(f"    Got response ({len(response)} chars, {thinking_tokens[0]} thinking tokens)")
        
        # Parse action
        action = parse_action(response, sample.goal)
        if verbose:
            print(f"    Action: {action.type} - questions={action.questions}, value={action.value}")
        
        if action.type == "ANSWER":
            result.final_answer = action.value
            result.answered = True
            result.correct = check_answer_correct(action.value, world.target_value, sample.goal)
            result.final_response = response
            result.turns_used = turn
            if verbose:
                print(f"    => ANSWER: {action.value}, correct={result.correct}")
            break
            
        elif action.type == "QUESTION":
            retry_count = 0  # Reset retry count on valid action
            
            # Get oracle answer
            oracle_resp = oracle_answer(clauses, base_context, world, action.questions)
            if verbose:
                print(f"    => Asked: {action.questions}, Oracle: {oracle_resp}")
            
            # Log this turn
            result.turn_logs.append(TurnLog(
                turn=turn,
                questions=action.questions,
                oracle_answer=oracle_resp,
                model_response=response,
            ))
            result.questions_asked.extend(action.questions)
            
            # Append to conversation
            if keep_thinking_trace and is_local_model(model_name):
                messages.append({"role": "assistant", "content": response})
            else:
                messages.append({"role": "assistant", "content": extract_non_thinking(response)})
            messages.append({"role": "user", "content": oracle_resp})
            
            turn += 1
            
        else:  # RETRY
            retry_count += 1
            if retry_count >= max_retries:
                result.final_response = response
                result.turns_used = turn
                result.budget_violated = True
                break
            
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": RETRY_PROMPT})
    
    if not result.answered:
        result.turns_used = turn
        result.budget_violated = True
    
    return result


async def run_episode_async(
    model_name: str,
    sample: Sample,
    world: World,
    budget: int,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    max_retries: int = 3,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
) -> EpisodeResult:
    """Async version of run_episode for parallel processing."""
    
    if verbose:
        print(f"\n--- Episode: sample={sample.sample_id}, world={world.assignments}, target={world.target_value} ---")
    
    # Build initial prompt
    messages = build_initial_prompt(sample, budget, emphasize_uncertainty=emphasize_uncertainty)
    clauses = parse_clauses(sample.rules)
    
    # Build base context from known facts
    base_context = {}
    for f in sample.known_facts:
        base_context[f] = True
    for f in sample.known_untrue_facts:
        base_context[f] = False
    
    result = EpisodeResult(
        sample_id=sample.sample_id,
        world=world,
    )
    
    retry_count = 0
    turn = 0
    
    while turn <= budget:
        if verbose:
            print(f"  Turn {turn}/{budget}...")
        
        # Force answer on last turn - use deep copy for thread safety
        current_messages = copy.deepcopy(messages)
        if turn == budget:
            current_messages[-1]["content"] += FORCE_ANSWER_PROMPT
        
        # Generate response using cached_generate (runs sync internally but is I/O bound)
        # For true async, we would use model_call_wrapper directly in async context
        # But cached_generate already handles batching and caching
        if verbose:
            print(f"    Generating response...")
        
        # Use asyncio.to_thread to run cached_generate in thread pool
        responses, thinking_tokens, cots, costs = await asyncio.to_thread(
            cached_generate,
            [current_messages],
            model_name,
            cache,
            cache_file,
            generation_config,
        )
        response = responses[0]
        if verbose:
            print(f"    Got response ({len(response)} chars, {thinking_tokens[0]} thinking tokens)")
        
        # Parse action
        action = parse_action(response, sample.goal)
        if verbose:
            print(f"    Action: {action.type} - questions={action.questions}, value={action.value}")
        
        if action.type == "ANSWER":
            result.final_answer = action.value
            result.answered = True
            result.correct = check_answer_correct(action.value, world.target_value, sample.goal)
            result.final_response = response
            result.turns_used = turn
            if verbose:
                print(f"    => ANSWER: {action.value}, correct={result.correct}")
            break
            
        elif action.type == "QUESTION":
            retry_count = 0  # Reset retry count on valid action
            
            # Get oracle answer
            oracle_resp = oracle_answer(clauses, base_context, world, action.questions)
            if verbose:
                print(f"    => Asked: {action.questions}, Oracle: {oracle_resp}")
            
            # Log this turn
            result.turn_logs.append(TurnLog(
                turn=turn,
                questions=action.questions,
                oracle_answer=oracle_resp,
                model_response=response,
            ))
            result.questions_asked.extend(action.questions)
            
            # Append to conversation
            if keep_thinking_trace and is_local_model(model_name):
                messages.append({"role": "assistant", "content": response})
            else:
                messages.append({"role": "assistant", "content": extract_non_thinking(response)})
            messages.append({"role": "user", "content": oracle_resp})
            
            turn += 1
            
        else:  # RETRY
            retry_count += 1
            if retry_count >= max_retries:
                result.final_response = response
                result.turns_used = turn
                result.budget_violated = True
                break
            
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": RETRY_PROMPT})
    
    if not result.answered:
        result.turns_used = turn
        result.budget_violated = True
    
    return result


async def process_episodes_async(
    episodes: List[Tuple[Sample, World]],
    model_name: str,
    budget: int,
    cache: Optional[Dict],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    keep_thinking_trace: bool = False,
    verbose: bool = False,
    emphasize_uncertainty: bool = True,
    max_concurrent: int = 8,
) -> List[EpisodeResult]:
    """Process multiple episodes concurrently with semaphore-based concurrency control."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(sample: Sample, world: World) -> EpisodeResult:
        async with semaphore:
            try:
                return await run_episode_async(
                    model_name=model_name,
                    sample=sample,
                    world=world,
                    budget=budget,
                    cache=cache,
                    cache_file=cache_file,
                    generation_config=generation_config,
                    keep_thinking_trace=keep_thinking_trace,
                    verbose=verbose,
                    emphasize_uncertainty=emphasize_uncertainty,
                )
            except Exception as e:
                print(f"Error processing episode sample={sample.sample_id}, world={world.assignments}: {e}")
                # Return a failed result
                result = EpisodeResult(sample_id=sample.sample_id, world=world)
                result.budget_violated = True
                return result
    
    # Create all tasks
    tasks = [process_with_semaphore(sample, world) for sample, world in episodes]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return list(results)


# ============ Metrics ============

def compute_minset_f1(asked: List[str], gt_sets: List[List[str]]) -> float:
    """Compute max F1 score vs any ground truth minimal set."""
    if not asked or not gt_sets:
        return 0.0
    
    asked_set = set(q.lower() for q in asked)
    max_f1 = 0.0
    
    for gt_set in gt_sets:
        gt_set_lower = set(q.lower() for q in gt_set)
        intersection = len(asked_set & gt_set_lower)
        if intersection > 0:
            precision = intersection / len(asked_set)
            recall = intersection / len(gt_set_lower)
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_metrics(results: List[EpisodeResult], samples: List[Sample]) -> Dict[str, float]:
    """Compute aggregate metrics."""
    if not results:
        return {}
    
    # Basic metrics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    answered = sum(1 for r in results if r.answered)
    
    # Micro accuracy
    micro_accuracy = correct / total if total > 0 else 0.0
    
    # Macro accuracy (per sample, average across worlds)
    sample_accuracies = {}
    for r in results:
        if r.sample_id not in sample_accuracies:
            sample_accuracies[r.sample_id] = []
        sample_accuracies[r.sample_id].append(1 if r.correct else 0)
    
    macro_accuracy = sum(
        sum(accs) / len(accs) for accs in sample_accuracies.values()
    ) / len(sample_accuracies) if sample_accuracies else 0.0
    
    # Questions used
    total_questions = sum(len(r.questions_asked) for r in results)
    avg_questions = total_questions / total if total > 0 else 0.0
    
    # MinSet F1
    sample_map = {s.sample_id: s for s in samples}
    minset_f1_scores = []
    for r in results:
        if r.sample_id in sample_map:
            f1 = compute_minset_f1(r.questions_asked, sample_map[r.sample_id].gt_min_sets)
            minset_f1_scores.append(f1)
    avg_minset_f1 = sum(minset_f1_scores) / len(minset_f1_scores) if minset_f1_scores else 0.0
    
    return {
        "total_episodes": total,
        "micro_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "answer_rate": answered / total if total > 0 else 0.0,
        "avg_questions_used": avg_questions,
        "avg_minset_f1": avg_minset_f1,
    }


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Multi-turn Logic-Q evaluator")
    parser.add_argument("--model_name", type=str, default="qwen_30b",
                        help="Model name (qwen_30b, gpt-5, gemini-2.5-flash, etc.)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to data CSV file")
    parser.add_argument("--budget", type=int, default=2,
                        help="Maximum number of questions allowed")
    parser.add_argument("--keep_thinking_trace", action="store_true",
                        help="Keep thinking trace in conversation (local models only)")
    parser.add_argument("--results_dir", type=str, default="results/multiturn",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging per episode")
    parser.add_argument("--no-emphasize-uncertainty", dest="emphasize_uncertainty",
                        action="store_false", default=True,
                        help="Disable the prompt emphasis that tells model to ask questions first")
    parser.add_argument("--async", dest="use_async", action="store_true", default=False,
                        help="Use async parallel processing for episodes")
    parser.add_argument("--max-concurrent", type=int, default=8,
                        help="Maximum concurrent episodes when using --async")
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    cache_dir = os.path.join(args.results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Setup cache
    data_basename = os.path.splitext(os.path.basename(args.data_file))[0]
    model_name_safe = args.model_name.replace("/", "_")
    output_name = f"{model_name_safe}-{data_basename}-budget{args.budget}"
    if args.keep_thinking_trace:
        output_name += "-keepthink"
    cache_file = os.path.join(cache_dir, f"{output_name}.jsonl")
    cache = load_cache_file(cache_file) if os.path.exists(cache_file) else {}
    
    # Generation config
    generation_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 32768,  # Increased as requested
    }
    
    # Handle model name aliases
    model_name = args.model_name
    if model_name == "qwen_30b":
        model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    elif model_name == "qwen_4b":
        model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"
    
    # Load data
    print(f"Loading data from {args.data_file}")
    samples = load_data(args.data_file)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples")
    
    # Count total episodes
    total_episodes = sum(len(s.worlds) for s in samples)
    print(f"Total episodes (samples × worlds): {total_episodes}")
    
    # Build episode list
    episodes = [(sample, world) for sample in samples for world in sample.worlds]
    
    # Run evaluation
    if args.use_async:
        print(f"Running async with max_concurrent={args.max_concurrent}")
        all_results = asyncio.run(
            process_episodes_async(
                episodes=episodes,
                model_name=model_name,
                budget=args.budget,
                cache=cache,
                cache_file=cache_file,
                generation_config=generation_config,
                keep_thinking_trace=args.keep_thinking_trace,
                verbose=args.verbose,
                emphasize_uncertainty=args.emphasize_uncertainty,
                max_concurrent=args.max_concurrent,
            )
        )
    else:
        # Sequential processing (original behavior)
        all_results = []
        pbar = tqdm.tqdm(total=total_episodes, desc="Running episodes")
        
        for sample in samples:
            for world in sample.worlds:
                result = run_episode(
                    model_name=model_name,
                    sample=sample,
                    world=world,
                    budget=args.budget,
                    cache=cache,
                    cache_file=cache_file,
                    generation_config=generation_config,
                    keep_thinking_trace=args.keep_thinking_trace,
                    verbose=args.verbose,
                    emphasize_uncertainty=args.emphasize_uncertainty,
                )
                all_results.append(result)
                pbar.update(1)
        
        pbar.close()
    
    # Compute metrics
    metrics = compute_metrics(all_results, samples)
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    results_file = os.path.join(args.results_dir, f"{output_name}_results.json")
    
    # Convert results to serializable format
    results_data = {
        "config": vars(args),
        "metrics": metrics,
        "episodes": [
            {
                "sample_id": r.sample_id,
                "world": r.world.assignments,
                "target_gt": r.world.target_value,
                "final_answer": r.final_answer,
                "correct": r.correct,
                "answered": r.answered,
                "turns_used": r.turns_used,
                "questions_asked": r.questions_asked,
                "turn_logs": [
                    {
                        "turn": t.turn,
                        "questions": t.questions,
                        "oracle_answer": t.oracle_answer,
                    }
                    for t in r.turn_logs
                ],
            }
            for r in all_results
        ],
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
