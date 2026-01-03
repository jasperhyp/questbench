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

"""Evaluate LLMs on GSM-Q (new version with k-question selection)."""

import json
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import datasets
from evaluators.evaluator import Evaluator
from model_utils import cached_generate
import pandas as pd
import tqdm

import hashlib
import ast

def _stable_seed(s: str) -> int:
  if s is None:
    s = ""
  h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
  return int(h[:8], 16)

def _safe_list_field(x: Any) -> List[Any]:
  if x is None:
    return []
  if isinstance(x, list):
    return x
  if isinstance(x, (set, tuple)):
    return list(x)
  if isinstance(x, (dict,)):
    return list(x.keys())
  if not isinstance(x, str):
    return []
  s = x.strip()
  if not s:
    return []
  try:
    return json.loads(s)
  except Exception:
    try:
      return ast.literal_eval(s)
    except Exception:
      return []

def _safe_json_loads(x: Any, default):
  if x is None:
    return default
  if isinstance(x, (list, dict)):
    return x
  try:
    return json.loads(x)
  except Exception:
    return default


def _infer_k_from_row(datum: pd.Series) -> int:
  for key in ["k", "Missing_Count", "missing_count", "missing_cnt"]:
    if key in datum and pd.notna(datum[key]):
      try:
        k = int(datum[key])
        return max(1, k)
      except Exception:
        pass
  return 1


def _parse_gt_question_field(gtq: Any) -> List[str]:
  """
  GT Question field may be:
  - a single variable name, e.g. "T"
  - a json list string, e.g. '["T","IR"]'
  - a python-ish list string with single quotes, e.g. "['T','IR']" (rare)
  Return: list of variable names
  """
  if gtq is None or (isinstance(gtq, float) and pd.isna(gtq)):
    return []
  if isinstance(gtq, list):
    return [str(x) for x in gtq]
  if not isinstance(gtq, str):
    return [str(gtq)]

  s = gtq.strip()
  if not s:
    return []
  # try json list
  if s.startswith("[") and s.endswith("]"):
    parsed = _safe_json_loads(s.replace("'", '"'), None)
    if isinstance(parsed, list):
      return [str(x) for x in parsed]
  return [s]


class GSMEvaluator(Evaluator):
  """Evaluator for LLMs on GSM-Q (k-question version).

  This version supports mc mode where the model can choose up to k questions
  (as a set of indices) from the predefined "Possible questions" list.
  """

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      verbal_questions: bool = False,
      eval_mode: str = "mc",
      batch_size: int = 1,
      **kwargs,
  ):
    super().__init__(
        model_name,
        cache=cache,
        cache_file=cache_file,
        use_cot=use_cot,
        fs_samples=fs_samples,
        eval_mode=eval_mode,
        **kwargs,
    )
    self.verbal_questions = verbal_questions

    # mc prompts: now parameterized by k
    self.assist_mc_prompt_k = """You are trying to solve a math problem. You must decide whether you have enough information to solve the math problem.
If you do not have enough information, you may ask questions back to the user from the predefined "Possible questions". Otherwise, choose "No questions needed."
Output format:
- If you want to ask: "Choice: <number_1>, <number_2>, ..." (all questions needed to solve this problem)
- If no questions needed: choose the option that says "No questions needed." and output its number as "Choice: <number>"
"""

    self.assist_isambig_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question, or "Not sure" if you are not sure what the answer is. Only include the raw numerical answer, do not include any units or thousands separators."""
    self.assist_fullinfo_prompt = """You are trying to answer a math question. Please answer with "Answer:" followed by the answer to the math question. Only include the raw numerical answer, do not include any units or thousands separators."""

    self.user_mc_prompt = """Math problem: {request}

Possible questions:
{possible_qs}"""
    self.user_isambig_prompt = """Math problem: {request}"""
    self.user_fullinfo_prompt = """Math problem: {request}"""

    if self.eval_mode == "mc":
      self.user_prompt = self.user_mc_prompt
      # system prompt will be chosen per-example based on k in make_batches
      self.assist_prompt = None
    elif self.eval_mode == "isambig":
      self.assist_prompt = self.assist_isambig_prompt
      self.user_prompt = self.user_isambig_prompt
    else:
      assert self.eval_mode == "fullinfo"
      self.assist_prompt = self.assist_fullinfo_prompt
      self.user_prompt = self.user_fullinfo_prompt

    if self.eval_mode != "mc":
      if self.use_cot:
        self.assist_prompt += " Reason step-by-step, then generate one of the above outputs."
      else:
        self.assist_prompt += " Generate one of the above outputs and nothing else."

    self.batch_size = batch_size
    self.orig_dataset = datasets.load_dataset("qintongli/GSM-Plus")

  def _build_mc_system_prompt(self, k: int) -> str:
    p = self.assist_mc_prompt_k

    if self.use_cot:
      p += " Reason step-by-step, then generate one of the above outputs."
    else:
      p += " Generate one of the above outputs and nothing else."
    return p

  def _parse_choice_set(self, response: str) -> Optional[Set[int]]:
    """
    Parse "Choice: 1, 2" -> {1,2}
    Also accept if model includes extra text, as long as digits exist after Choice.
    """
    if response is None:
      return None
    text = response.strip()
    lower = text.lower()
    if "choice" in lower:
      # take substring after last "choice:"
      lower = lower.split("choice:")[-1]
      text_after = lower
    else:
      text_after = lower

    nums = re.findall(r"\b[0-9]+\b", text_after)
    if not nums:
      return None
    try:
      return set(int(x) for x in nums)
    except Exception:
      return None

  def _needs_retry(self, response: str) -> bool:
    if self.eval_mode == "mc":
      return self._parse_choice_set(response) is None
    else:
      return not re.findall(r"(not sure|\b[0-9]+\b)", (response or "").lower())

  def evaluate_batch(
      self,
      batch_requests: List[str],
      batch_system_prompts: List[Optional[str]],
      batch_gt_queries,
      model_name: str,
      model_url: str,
      cache=None,
      cache_file=None,
      fs_turns=None,
  ):
    """
    Returns:
      batch_convos, batch_preds, batch_correct, cost, all_cots
    """
    batch_prompts = []
    for request, system_prompt in zip(batch_requests, batch_system_prompts):
      assist_prompt = []
      if self.fs_samples > 0 and fs_turns is not None:
        assist_prompt.extend(fs_turns)
      if system_prompt is None:
        assist_prompt.append({"role": "user", "content": request})
      else:
        assist_prompt.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ])
      batch_prompts.append(assist_prompt)

    batch_responses, cost, all_cots = cached_generate(
        batch_prompts,
        model_name,
        model_url,
        cache=cache,
        cache_file=cache_file,
        generation_config=self.generation_config,
        parallel_model_calls=self.parallel_model_calls,
    )

    # Build initial conversations from prompts
    batch_convos = []
    for i, prompt in enumerate(batch_prompts):
      convo = []
      for msg in prompt:
        convo.append({"role": msg["role"], "text": msg["content"]})
      convo.append({"role": self.model_role_name, "text": batch_responses[i]})
      batch_convos.append(convo)

    # Batched retry loop
    max_retry_rounds = 5
    for retry_round in range(max_retry_rounds):
      retry_indices = []
      retry_prompts = []
      retry_messages = []

      for i, resp in enumerate(batch_responses):
        if self._needs_retry(resp):
          retry_indices.append(i)

          # append failed response then corrective instruction
          batch_prompts[i].append({"role": self.model_role_name, "content": resp})

          if self.eval_mode == "mc":
            retry_msg = (
                'Wrong format or option not found. Output exactly "Choice: <number>" '
                'or "Choice: <number_1>, <number_2>, ..." and nothing else.'
            )
          elif self.eval_mode == "fullinfo":
            retry_msg = (
                'Wrong format. Output exactly "Answer: <number>" (raw number only) and nothing else.'
            )
          else:
            retry_msg = (
                'Wrong format. Output exactly "Answer: <number>" or "Answer: Not sure" and nothing else.'
            )

          batch_prompts[i].append({"role": "user", "content": retry_msg})
          retry_prompts.append(batch_prompts[i])
          retry_messages.append(retry_msg)

      if not retry_indices:
        break

      if retry_round == max_retry_rounds - 1:
        print(f"Max retries reached for {len(retry_indices)} responses")
        break

      retry_responses, retry_cost, retry_cots = cached_generate(
          retry_prompts,
          model_name,
          model_url,
          cache=cache,
          cache_file=cache_file,
          generation_config=self.generation_config,
          parallel_model_calls=self.parallel_model_calls,
      )
      
      # SANITY: random guesses, should be near 0 acc on missing samples
      # rand = random.Random(0)
      # for i in range(len(batch_responses)):
      #   # guess a single option 0..5
      #   batch_responses[i] = f"Choice: {rand.randint(0, 5)}"

      for idx, orig_i in enumerate(retry_indices):
        batch_responses[orig_i] = retry_responses[idx]
        all_cots[orig_i] = retry_cots[idx]
        cost[orig_i] = retry_cost[idx]

        batch_convos[orig_i].append({"role": "user", "text": retry_messages[idx]})
        batch_convos[orig_i].append({"role": self.model_role_name, "text": retry_responses[idx]})

    # Parse final responses & correctness
    batch_preds = []
    batch_correct = []
    for i, resp in enumerate(batch_responses):
      if self.eval_mode == "mc":
        pred_set = self._parse_choice_set(resp)
        if pred_set is None:
          pred_set = set()
        batch_preds.append(pred_set)

        # gt can be int, list[int], list[list[int]], etc.
        gt = batch_gt_queries[i]

        # Normalize gt to a list of sets
        gt_sets = []
        if isinstance(gt, int):
          gt_sets = [set([gt])]
        elif isinstance(gt, (set, frozenset)):
          gt_sets = [set(gt)]
        elif isinstance(gt, list):
          # could be list[int] (single set), or list[list[int]] (multiple acceptable)
          if len(gt) == 0:
            gt_sets = [set()]
          elif all(isinstance(x, int) for x in gt):
            gt_sets = [set(gt)]
          elif all(isinstance(x, (list, set, tuple)) for x in gt):
            for x in gt:
              gt_sets.append(set(x))
          else:
            # fallback: try convert numeric-like
            tmp = []
            for x in gt:
              try:
                tmp.append(int(x))
              except Exception:
                pass
            gt_sets = [set(tmp)]
        else:
          # fallback
          try:
            gt_sets = [set([int(gt)])]
          except Exception:
            gt_sets = [set()]

        is_match = any(pred_set == s for s in gt_sets)
        batch_correct.append(is_match)

      else:
        # isambig/fullinfo: parse Answer
        low = (resp or "").lower()
        answer_part = low.split("answer:")[-1].strip()

        if "not sure" in answer_part:
          pred = "Not sure"
          is_match = (batch_gt_queries[i] == "Not sure")
        else:
          nums = re.findall(r"\b[0-9]+\b", answer_part)
          if not nums:
            pred = "None"
            is_match = False
          else:
            pred = nums[0]
            try:
              is_match = int(pred) == int(batch_gt_queries[i])
            except Exception:
              is_match = False

        batch_preds.append(pred)
        batch_correct.append(is_match)

    return batch_convos, batch_preds, batch_correct, cost, all_cots

  def make_convo_batches(self, data: pd.DataFrame, batch_size: Optional[int] = None):
    if batch_size is None:
      batch_size = self.batch_size

    batch_ids = [[]]
    batch_requests = [[]]
    batch_gt_answers = [[]]
    batch_gt_queries = [[]]
    batch_system_prompts = [[]]
    batch_k = [[]]

    for d, (_, datum) in enumerate(data.iterrows()):
      if self.eval_mode == "mc":
        k = _infer_k_from_row(datum)
        request = datum["Rewritten Problem"]

        variables = _safe_json_loads(datum.get("Variables", "{}"), {})
        possible_qs = _safe_list_field(datum.get("Possible Questions", "[]"))
        gt_vars = _safe_list_field(datum.get("Heldout Value", "[]"))

        paired = [(str(v), str(v) in set(str(x) for x in gt_vars)) for v in possible_qs]

        uid = str(datum.get("unique_id", datum.get("Question ID", d)))
        rng = random.Random(_stable_seed(uid))
        rng.shuffle(paired)

        questions = []
        var_to_index = {}
        gt_indices = []

        for idx, (var, is_gt) in enumerate(paired):
            var_to_index[var] = idx
            if is_gt:
                gt_indices.append(idx)

            if self.verbal_questions:
                questions.append(f"{idx}. What is {variables.get(var, var)} ({var})?")
            else:
                questions.append(f"{idx}. What is the value of {var}?")

        no_q_index = len(questions)
        questions.append(f"{no_q_index}. No questions needed.")
        #######

        # # ground truth is a set of indices (support multi-var if provided)
        # if len(gt_vars) == 0:
        #   gt_set = set([no_q_index])
        # else:
        #   idxs = []
        #   for var in gt_vars:
        #     if var in var_to_index:
        #       idxs.append(var_to_index[var])
        #   if len(idxs) == 0:
        #     gt_set = set([no_q_index])
        #   else:
        #     gt_set = set(idxs)
        
        k_in_row = _infer_k_from_row(datum)
        if k_in_row == 0 or len(gt_vars) == 0:
          gt_set = set([no_q_index])
        else:
          idxs = []
          for var in gt_vars:
            if str(var) in var_to_index:
              idxs.append(var_to_index[str(var)])
          gt_set = set(idxs) if len(idxs) > 0 else set([no_q_index])

        answer = datum.get("Full Answer", None)

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])
          batch_system_prompts.append([])
          batch_k.append([])

        batch_ids[-1].append(d)
        batch_requests[-1].append(
            self.user_prompt.format(
                request=request,
                possible_qs="\n".join(questions),
            )
        )
        batch_gt_answers[-1].append(answer)
        batch_gt_queries[-1].append(list(gt_set))
        batch_system_prompts[-1].append(self._build_mc_system_prompt(k))
        batch_k[-1].append(k)

      else:
        is_trues = [True]
        if self.eval_mode == "isambig":
          is_trues = [True, None]

        for is_true in is_trues:
          if is_true is None:
            request = datum["Rewritten Problem"]
            response = "Not sure"
          else:
            if self.verbal_questions:
              request = self.orig_dataset["test"]["question"][datum["Question ID"]]
            else:
              request = datum["Full Problem"]
            response = datum["Full Answer"]

          if len(batch_requests[-1]) >= batch_size:
            batch_ids.append([])
            batch_requests.append([])
            batch_gt_answers.append([])
            batch_gt_queries.append([])
            batch_system_prompts.append([])
            batch_k.append([])

          batch_ids[-1].append(d)
          batch_requests[-1].append(self.user_prompt.format(request=request))
          batch_gt_queries[-1].append(response)
          batch_gt_answers[-1].append(datum["Full Answer"])
          batch_system_prompts[-1].append(self.assist_prompt)
          batch_k[-1].append(_infer_k_from_row(datum))

    return batch_ids, batch_system_prompts, batch_requests, batch_gt_answers, batch_gt_queries, batch_k

  def make_fewshot_turns(self, fewshot_data: pd.DataFrame):
    fewshot_turns = []
    if fewshot_data is None:
      return []

    for d, (_, datum) in enumerate(fewshot_data.iterrows()):
      if d >= self.fs_samples:
        break

      if self.eval_mode == "mc":
        # few-shot for mc: choose k from datum if present
        k = _infer_k_from_row(datum)

        request = datum["Rewritten Problem"]
        variables = _safe_json_loads(datum.get("Variables", "{}"), {})
        # possible_qs = _safe_json_loads(datum.get("Possible Questions", "[]"), [])
        possible_qs = json.loads(datum["Possible Questions"])
        q_to_ask = datum["GT Question"]
        gt_vars = _parse_gt_question_field(datum.get("GT Question", ""))

        questions = []
        var_to_index = {}
        for v, variable in enumerate(possible_qs):
          var_to_index[str(variable)] = v
          if self.verbal_questions:
            questions.append(f"{v}. What is {variables.get(variable, variable)} ({variable})?")
          else:
            questions.append(f"{v}. What is the value of {variable}?")
        no_q_index = len(questions)
        questions.append(f"{no_q_index}. No questions needed.")

        idxs = []
        for var in gt_vars:
          if var in var_to_index:
            idxs.append(var_to_index[var])
        if len(idxs) == 0:
          idxs = [no_q_index]
        # keep at most k in few-shot label
        idxs = idxs[: max(1, k)]

        fewshot_turns.append([
            {
                "role": "system",
                "content": self._build_mc_system_prompt(k),
            },
            {
                "role": "user",
                "content": self.user_prompt.format(
                    request=request,
                    possible_qs="\n".join(questions),
                ),
            },
            {
                "role": self.model_role_name,
                "content": "Choice: " + ", ".join(str(x) for x in idxs),
            },
        ])

      else:
        is_trues = [True]
        if self.eval_mode == "isambig":
          is_trues = [True, None]
        is_true = is_trues[len(fewshot_turns) % len(is_trues)]

        if is_true is None:
          request = datum["Rewritten Problem"]
          response = "Not sure"
        else:
          if self.verbal_questions:
            request = self.orig_dataset["test"]["question"][datum["Question ID"]]
          else:
            request = datum["Full Problem"]
          response = datum["Full Answer"]

        fewshot_turns.append([
            {"role": "system", "content": self.assist_prompt},
            {"role": "user", "content": self.user_prompt.format(request=request)},
            {"role": self.model_role_name, "content": f"Answer: {response}"},
        ])

    random.shuffle(fewshot_turns)
    fewshot_prefix = []
    for pair in fewshot_turns:
      for turn in pair:
        fewshot_prefix.append(turn)

    return fewshot_prefix

  def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
    results = pd.DataFrame(
        columns=[
            "k",
            "correct",
            "max_depth",
            "pred_answer",
            "gt_answer",
            "id",
            "request",
            "CSP",
            "num_constraints",
            "num_vars",
            "pred_q",
            "gt_qs",
            "all_qs",
            "conversation",
        ]
    )

    fs_turns = self.make_fewshot_turns(prompt_data)

    (
        batch_ids,
        batch_system_prompts,
        batch_requests,
        batch_gt_answers,
        batch_gt_queries,
        batch_k,
    ) = self.make_convo_batches(data)

    total_cost = []
    all_cots = []

    pbar = tqdm.tqdm(
        zip(batch_ids, batch_system_prompts, batch_requests, batch_gt_answers, batch_gt_queries, batch_k),
        total=len(batch_ids),
    )

    for batch_id, batch_system_prompt, batch_request, batch_gt_answer, batch_gt_query, batch_k_vals in pbar:
      batch_conversation, batch_pred, batch_correct, cost, cots = self.evaluate_batch(
          batch_request,
          batch_system_prompt,
          batch_gt_query,
          model_name=self.model_name,
          model_url=self.model_url,
          cache=self.cache,
          cache_file=self.cache_file,
          fs_turns=fs_turns,
      )

      total_cost += cost
      all_cots += cots

      for i, item_id in enumerate(batch_id):
        datum = data.iloc[item_id]
        equations = _safe_json_loads(datum.get("Equations", "{}"), {})
        variables = _safe_json_loads(datum.get("Variables", "{}"), {})

        pred_answer = None

        results.loc[len(results)] = [
            batch_k_vals[i],
            batch_correct[i],
            datum.get("depth", datum.get("max_depth", None)),
            pred_answer,
            batch_gt_answer[i],
            item_id,
            batch_request[i],
            datum.get("CSP", None),
            len(equations) if isinstance(equations, dict) else (len(equations) if isinstance(equations, list) else None),
            len(variables) if isinstance(variables, dict) else None,
            batch_pred[i],
            batch_gt_query[i],
            variables,
            json.dumps(batch_conversation[i]),
        ]

      results_filtered = results[results["correct"].notna()]
      pbar.set_description(f"Accuracy: {sum(results_filtered['correct']) / len(results_filtered)}")

    results_filtered = results[results["correct"].notna()]
    print(f"Final accuracy: {sum(results_filtered['correct']) / len(results_filtered)}")
    if "max_depth" in results.columns:
      try:
        print("Accuracy by depth:", results.groupby("max_depth").agg({"correct": "mean"}))
      except Exception:
        pass

    print(f"Total cost: {total_cost}")
    return results, all_cots, total_cost
