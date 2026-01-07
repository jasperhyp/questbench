"""Evaluate LLMs on GSM-Q.

Compatibility note:
This file keeps the original gsm.py behavior, but now also returns gsm_new-style
(cost, cots) outputs:
  evaluate_data(...) -> (results_df, all_cots, total_cost_list)

It writes per-example accumulated USD cost into results_df["cost"].
It also writes per-example accumulated thinking tokens into results_df["thinking_tokens"].

This version is adapted to the current model_utils.cached_generate() which returns:
  texts: List[str]
  thinking_tokens: List[Optional[int]]
  cots: List[Optional[str]]
  costs_usd: List[Optional[float]]
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, List, Set, Tuple

import datasets  # kept for compatibility, may be unused
import pandas as pd
import tqdm

from evaluators.evaluator import Evaluator
from model_utils import cached_generate


def _safe_list_field(x):
  if x is None:
    return []
  if isinstance(x, list):
    return x
  if isinstance(x, (set, tuple)):
    return list(x)
  if isinstance(x, dict):
    return list(x.keys())
  if not isinstance(x, str):
    return []
  s = x.strip()
  if not s:
    return []
  try:
    return json.loads(s)
  except Exception:
    return []


def _parse_choice_int(response: str, valid_set: Set[int]):
  if response is None:
    return None
  text = response.strip().lower()
  if "choice:" in text:
    text = text.split("choice:")[-1]
  nums = re.findall(r"\b[0-9]+\b", text)
  if len(nums) != 1:
    return None
  try:
    v = int(nums[0])
  except Exception:
    return None
  if v not in valid_set:
    return None
  return v


def _as_cost_usd_list(costs_usd: Any, n: int) -> List[float]:
  if n <= 0:
    return []
  if costs_usd is None:
    return [0.0] * n
  if isinstance(costs_usd, list):
    out: List[float] = []
    for i in range(n):
      try:
        v = costs_usd[i]
      except Exception:
        v = None
      try:
        out.append(float(v) if v is not None else 0.0)
      except Exception:
        out.append(0.0)
    return out
  try:
    v = float(costs_usd)
    return [v] * n
  except Exception:
    return [0.0] * n


def _as_thinking_list(thinking_tokens: Any, n: int) -> List[int]:
  if n <= 0:
    return []
  if thinking_tokens is None:
    return [0] * n
  if isinstance(thinking_tokens, list):
    out: List[int] = []
    for i in range(n):
      try:
        v = thinking_tokens[i]
      except Exception:
        v = None
      try:
        out.append(int(v) if v is not None else 0)
      except Exception:
        out.append(0)
    return out
  try:
    v = int(thinking_tokens)
    return [v] * n
  except Exception:
    return [0] * n


def _first_or_none(x: Any):
  if x is None:
    return None
  if isinstance(x, list):
    return x[0] if len(x) > 0 else None
  return x


class GSMEvaluator(Evaluator):
  """Evaluator for LLMs on GSM-Q."""

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
      reveal_k_in_prompt: bool = False,
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
    self.reveal_k_in_prompt = reveal_k_in_prompt

    self.assist_mc_prompt = """You are trying to solve a math problem. You must decide whether you have enough information to solve the math problem. Please respond with one of the following-
If you do not have enough information to solve the math problem, you may ask a question back to the user from a set of predefined "Possible questions". Otherwise, choose "No questions needed."
Generate the number of your choice in the form "Choice: number"
"""

    self.assist_mc_prompt_k = """You are trying to solve a math problem. You must decide whether you have enough information to solve the math problem.
If you do not have enough information, you may ask questions back to the user from the predefined "Possible questions". Otherwise, choose "No questions needed."
Output format:
- If you want to ask: "Choice: <number_1>, <number_2>, ..." (all questions needed to solve this problem)
- If no questions needed: choose the option that says "No questions needed." and output its number as "Choice: <number>"
"""

    self.assist_mc_prompt_give_k = """You are solving a math problem. Exactly {k} key variables required to solve the problem are missing from the given information.
Your task is to identify which {k} missing variables they are by selecting the corresponding questions from the predefined "Possible questions" list.
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

    self.assist_sc_prompt = """You are solving a math problem.
Decide how many key variables required to solve the problem are missing from the given information.
Choose exactly one option from {0,1,2,3,4}.
Output format:
Choice: <0-4>
"""
    self.user_sc_prompt = """Math problem: {request}"""

    if self.eval_mode == "mc":
      if self.reveal_k_in_prompt:
        self.assist_prompt = self.assist_mc_prompt_give_k.format(k=int(1))
      else:
        self.assist_prompt = self.assist_mc_prompt_k
      self.user_prompt = self.user_mc_prompt
    elif self.eval_mode == "sc":
      self.assist_prompt = self.assist_sc_prompt
      self.user_prompt = self.user_sc_prompt
    elif self.eval_mode == "isambig":
      self.assist_prompt = self.assist_isambig_prompt
      self.user_prompt = self.user_isambig_prompt
    else:
      assert self.eval_mode == "fullinfo"
      self.assist_prompt = self.assist_fullinfo_prompt
      self.user_prompt = self.user_fullinfo_prompt

    if self.use_cot and self.eval_mode in ["isambig", "fullinfo"]:
      self.assist_prompt += " Reason step-by-step, then generate one of the above outputs."
    else:
      self.assist_prompt += " Generate one of the above outputs and nothing else."

    self.batch_size = batch_size

  def generate_query(
      self,
      request,
      gt_query,
      fs_turns,
  ) -> Tuple[bool, str, list, float, int, Any]:
    """
    Returns:
      correct, pred_str, conversation, total_cost_usd, total_thinking_tokens, last_cot
    """
    conversation = []
    total_thinking = 0

    prompt = [
        {"role": "system", "content": self.assist_prompt},
        *fs_turns,
        {"role": "user", "content": request},
    ]

    texts, thinking0, cots0, costs0 = cached_generate(
        [prompt],
        self.model_name,
        self.model_url,
        cache=self.cache,
        cache_file=self.cache_file,
        generation_config=self.generation_config,
        parallel_model_calls=False,
    )
    response = texts[0] if texts else ""
    total_cost = _as_cost_usd_list(costs0, 1)[0]
    total_thinking += _as_thinking_list(thinking0, 1)[0]
    last_cot = _first_or_none(cots0)

    conversation.append({"role": "user", "text": request})
    conversation.append({"role": self.model_role_name, "text": response})
    prompt.append({"role": self.model_role_name, "content": response})

    if self.eval_mode == "mc":
      resp_text = response.lower().split("choice:")[-1].strip()
      n_loops = 0
      while not re.findall(r"\b[0-9]+\b", resp_text):
        retry_msg = (
            "Wrong format or option not found. Please provide the number of"
            ' your choice. Output "Choice: <number>" and nothing else.'
        )
        prompt.append({"role": "system", "content": retry_msg})

        texts_r, thinking_r, cots_r, costs_r = cached_generate(
            [prompt],
            self.model_name,
            self.model_url,
            cache=self.cache,
            cache_file=self.cache_file,
            generation_config=self.generation_config,
            parallel_model_calls=False,
        )
        response = texts_r[0] if texts_r else ""
        total_cost += _as_cost_usd_list(costs_r, 1)[0]
        total_thinking += _as_thinking_list(thinking_r, 1)[0]
        if cots_r is not None:
          last_cot = _first_or_none(cots_r)

        conversation.append({"role": "system", "content": retry_msg})
        conversation.append({"role": self.model_role_name, "text": response})
        prompt.append({"role": self.model_role_name, "content": response})

        resp_text = response.lower().split("choice:")[-1].strip()
        n_loops += 1
        if n_loops > 5:
          break

      try:
        picked = re.findall(r"\b[0-9]+\b", resp_text)[0]
        correct = int(picked) == int(gt_query)
      except Exception:
        picked = resp_text
        correct = False

      return correct, str(picked), conversation, float(total_cost), int(total_thinking), last_cot

    if self.eval_mode == "sc":
      valid = set([0, 1, 2, 3, 4])
      pred = _parse_choice_int(response, valid)
      n_loops = 0
      while pred is None:
        retry_msg = 'Wrong format. Output exactly "Choice: <0-4>" and nothing else.'
        prompt.append({"role": "system", "content": retry_msg})

        texts_r, thinking_r, cots_r, costs_r = cached_generate(
            [prompt],
            self.model_name,
            self.model_url,
            cache=self.cache,
            cache_file=self.cache_file,
            generation_config=self.generation_config,
            parallel_model_calls=False,
        )
        response = texts_r[0] if texts_r else ""
        total_cost += _as_cost_usd_list(costs_r, 1)[0]
        total_thinking += _as_thinking_list(thinking_r, 1)[0]
        if cots_r is not None:
          last_cot = _first_or_none(cots_r)

        conversation.append({"role": "system", "content": retry_msg})
        conversation.append({"role": self.model_role_name, "text": response})
        prompt.append({"role": self.model_role_name, "content": response})

        pred = _parse_choice_int(response, valid)
        n_loops += 1
        if n_loops > 5:
          break

      if pred is None:
        correct = False
        pred_out = "-1"
      else:
        pred_out = str(pred)
        try:
          correct = int(pred) == int(gt_query)
        except Exception:
          correct = False

      return correct, pred_out, conversation, float(total_cost), int(total_thinking), last_cot

    resp_text = response.lower().split("answer:")[-1].strip()
    n_loops = 0
    while not re.findall(r"(not sure|\b[0-9]+\b)", resp_text.lower()):
      if self.eval_mode == "fullinfo":
        prompt_content = (
            "Wrong format. Please provide the answer as a raw number without"
            " any additional units, thousands separators, or text. Output"
            ' "Answer: <number>" and nothing else.'
        )
      else:
        prompt_content = (
            'Wrong format. Please answer either "Answer: <number>" or'
            ' "Answer: Not sure" and nothing else. If you answer with a'
            " number, it should be a raw number without any additional units,"
            " thousands separators, or text."
        )
      prompt.append({"role": "system", "content": prompt_content})

      texts_r, thinking_r, cots_r, costs_r = cached_generate(
          [prompt],
          self.model_name,
          self.model_url,
          cache=self.cache,
          cache_file=self.cache_file,
          generation_config=self.generation_config,
          parallel_model_calls=False,
      )
      response = texts_r[0] if texts_r else ""
      total_cost += _as_cost_usd_list(costs_r, 1)[0]
      total_thinking += _as_thinking_list(thinking_r, 1)[0]
      if cots_r is not None:
        last_cot = _first_or_none(cots_r)

      conversation.append({"role": "system", "content": prompt_content})
      conversation.append({"role": self.model_role_name, "text": response})
      prompt.append({"role": self.model_role_name, "content": response})

      resp_text = response.lower().split("answer:")[-1].strip()
      n_loops += 1
      if n_loops > 5:
        break

    if "not sure" in resp_text.lower():
      correct = gt_query == "Not sure"
      pred_out = "Not sure"
    else:
      numbers = re.findall(r"\b[0-9]+\b", resp_text)
      if not numbers:
        correct = False
        pred_out = resp_text
      else:
        pred_out = numbers[0]
        try:
          correct = int(pred_out) == int(gt_query)
        except Exception:
          correct = False

    return correct, str(pred_out), conversation, float(total_cost), int(total_thinking), last_cot

  def generate_query_batch(self, batch_request, batch_gt_query, fs_turns):
    """
    Returns:
      batch_query, batch_conversation, batch_correct, batch_cost, batch_thinking, batch_cot
    """
    batch_query = []
    batch_conversation = []
    batch_correct = []
    batch_cost = []
    batch_thinking = []
    batch_cot = []

    for request, gt_query in zip(batch_request, batch_gt_query):
      correct, query, conversation, cost, thinking, cot = self.generate_query(
          request, gt_query, fs_turns
      )
      batch_query.append(query)
      batch_conversation.append(conversation)
      batch_correct.append(correct)
      batch_cost.append(cost)
      batch_thinking.append(thinking)
      batch_cot.append(cot)

    return batch_query, batch_conversation, batch_correct, batch_cost, batch_thinking, batch_cot

  def make_convo_batches(self, data, batch_size=None):
    if batch_size is None:
      batch_size = self.batch_size

    batch_ids = [[]]
    batch_requests = [[]]
    batch_gt_answers = [[]]
    batch_gt_queries = [[]]

    for d, (_, datum) in enumerate(data.iterrows()):
      if self.eval_mode == "mc":
        request = datum["Rewritten Problem"]
        variables = json.loads(datum["Variables"])
        possible_qs = json.loads(datum["Possible Questions"])
        q_to_ask = datum["GT Question"]

        questions = []
        q_to_ask_index = -1
        for v, variable in enumerate(possible_qs):
          if self.verbal_questions:
            questions.append(f"{v}. What is {variables[variable]} ({variable})?")
          else:
            questions.append(f"{v}. What is the value of {variable}?")
          if variable == q_to_ask:
            q_to_ask_index = v
        questions.append(f"{len(questions)}. No questions needed.")
        answer = datum["Full Answer"]

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])

        batch_requests[-1].append(
            self.user_prompt.format(
                request=request,
                possible_qs="\n".join(questions),
            )
        )
        batch_gt_queries[-1].append(q_to_ask_index)
        batch_ids[-1].append(d)
        batch_gt_answers[-1].append(answer)

      elif self.eval_mode == "sc":
        request = datum["Rewritten Problem"]
        missing_cnt = 1
        answer = datum.get("Full Answer", None)

        if len(batch_requests[-1]) >= batch_size:
          batch_ids.append([])
          batch_requests.append([])
          batch_gt_answers.append([])
          batch_gt_queries.append([])

        batch_requests[-1].append(self.user_prompt.format(request=request))
        batch_gt_queries[-1].append(int(missing_cnt))
        batch_ids[-1].append(d)
        batch_gt_answers[-1].append(answer)

      else:
        is_trues = [True]
        if self.eval_mode == "isambig":
          is_trues = [True, None]

        for is_true in is_trues:
          if is_true is None:
            request = datum["Rewritten Problem"]
            response = "Not sure"
          else:
            request = datum["Full Problem"]
            response = datum["Full Answer"]

          if len(batch_requests[-1]) >= batch_size:
            batch_ids.append([])
            batch_requests.append([])
            batch_gt_answers.append([])
            batch_gt_queries.append([])

          batch_ids[-1].append(d)
          batch_requests[-1].append(self.user_prompt.format(request=request))
          batch_gt_queries[-1].append(response)
          batch_gt_answers[-1].append(datum["Full Answer"])

    return batch_ids, batch_requests, batch_gt_answers, batch_gt_queries

  def make_fewshot_turns(self, fewshot_data):
    fewshot_turns = []
    if fewshot_data is None:
      return []

    for d, (_, datum) in enumerate(fewshot_data.iterrows()):
      if d >= self.fs_samples:
        break

      if self.eval_mode == "mc":
        request = datum["Rewritten Problem"]
        q_to_ask = datum["GT Question"]
        variables = json.loads(datum["Variables"])

        questions = []
        q_to_ask_index = -1
        possible_qs = json.loads(datum["Possible Questions"])
        for v, variable in enumerate(possible_qs):
          if self.verbal_questions:
            questions.append(f"{v}. What is {variables[variable]} ({variable})?")
          else:
            questions.append(f"{v}. What is the value of {variable}?")
          if variable == q_to_ask:
            q_to_ask_index = v
        questions.append(f"{len(questions)}. No questions needed.")

        fewshot_turns.append([
            {
                "role": "user",
                "content": self.user_prompt.format(
                    request=request,
                    possible_qs="\n".join(questions),
                ),
            },
            {
                "role": self.model_role_name,
                "content": f"Choice: {q_to_ask_index}",
            },
        ])

      elif self.eval_mode == "sc":
        request = datum["Rewritten Problem"]
        missing_cnt = 1
        fewshot_turns.append([
            {"role": "user", "content": self.user_prompt.format(request=request)},
            {"role": self.model_role_name, "content": f"Choice: {int(missing_cnt)}"},
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
          request = datum["Full Problem"]
          response = datum["Full Answer"]

        fewshot_turns.append([
            {"role": "user", "content": self.user_prompt.format(request=request)},
            {"role": self.model_role_name, "content": f"Answer: {response}"},
        ])

    random.shuffle(fewshot_turns)
    fewshot_prefix = []
    for sublist in fewshot_turns:
      for turn in sublist:
        fewshot_prefix.append(turn)
    return fewshot_prefix

  def evaluate_data(self, data: pd.DataFrame, prompt_data: pd.DataFrame):
    """
    Returns:
      (results_df, all_cots, total_think_tokens_list, all_cost_usd_list)
    """
    results = pd.DataFrame(
        columns=[
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
            "cost",              # per-example USD cost
            "thinking_tokens",   # per-example thinking tokens
        ]
    )

    fs_turns = self.make_fewshot_turns(prompt_data) if prompt_data is not None else []
    batch_ids, batch_requests, batch_gt_answers, batch_gt_queries = self.make_convo_batches(data)

    all_cost_usd: List[float] = []
    total_think_tokens: List[int] = []
    all_cots: List[Any] = []

    pbar = tqdm.tqdm(
        zip(batch_ids, batch_requests, batch_gt_answers, batch_gt_queries),
        total=len(batch_ids),
    )

    for batch_id, batch_request, batch_gt_answer, batch_gt_query in pbar:
      batch_query, batch_conversation, batch_correct, batch_cost, batch_thinking, batch_cot = (
          self.generate_query_batch(batch_request, batch_gt_query, fs_turns)
      )

      # aggregate
      all_cost_usd += [float(x) if x is not None else 0.0 for x in batch_cost]
      total_think_tokens += [int(x) if x is not None else 0 for x in batch_thinking]
      all_cots += batch_cot

      for i, item_id in enumerate(batch_id):
        datum = data.iloc[item_id]
        pred_answer = None
        equations = json.loads(datum["Equations"])
        variables = json.loads(datum["Variables"])

        results.loc[len(results)] = [
            batch_correct[i],
            datum["depth"],
            pred_answer,
            batch_gt_answer[i],
            batch_id[i],
            batch_request[i],
            datum["CSP"],
            len(equations),
            len(variables),
            batch_query[i],
            batch_gt_query[i],
            variables,
            json.dumps(batch_conversation[i]),
            float(batch_cost[i]) if batch_cost[i] is not None else 0.0,
            int(batch_thinking[i]) if batch_thinking[i] is not None else 0,
        ]

      results_filtered = results[results["correct"].notna()]
      if len(results_filtered) > 0:
        pbar.set_description(
            "Accuracy:"
            f" {sum(results_filtered['correct']) / len(results_filtered)}"
        )

    results_filtered = results[results["correct"].notna()]
    if len(results_filtered) > 0:
      print(
          "Final accuracy:"
          f" {sum(results_filtered['correct']) / len(results_filtered)}"
      )
      try:
        print(
            "Accuracy by depth:",
            results.groupby("max_depth").agg({"correct": "mean"}),
        )
      except Exception:
        pass

    return results, all_cots, total_think_tokens, all_cost_usd
