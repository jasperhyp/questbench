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

"""Utility functions for calling models.

This version unifies GPT and Gemini outputs to match Qwen's style:
  (final_output, num_thinking_tokens, cot)

cached_generate returns:
  text_outputs: List[str]
  thinking_tokens: List[Optional[int]]
  cots: List[Optional[str]]
  costs_usd: List[Optional[float]]
"""

from __future__ import annotations

from concurrent import futures
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import asyncio
from urllib.parse import urlparse

import httpx
from openai import OpenAI, AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
import requests
import tenacity
from tenacity import retry
import transformers


ThreadPoolExecutor = futures.ThreadPoolExecutor
pipeline = transformers.pipeline
wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt


# ----------------------------
# Headers and endpoints
# ----------------------------

OPENAI_HEADER: Dict[str, str] = {}
if "OPENAI_API_KEY" in os.environ:
  OPENAI_HEADER = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
  }
  # drop empty fields to avoid confusing some gateways
  OPENAI_HEADER = {k: v for k, v in OPENAI_HEADER.items() if v}

# Default OpenAI Chat Completions endpoint (can be overridden by evaluator's model_url)
OPENAI_CHAT_COMPLETIONS_URL = os.environ.get(
    "OPENAI_MODEL_URL",
    "https://api.openai.com/v1/chat/completions",
).strip()

# Gemini OpenAI compatibility endpoint (Developer API)
# You can override this if you proxy it, for example via API Gateway
GEMINI_MODEL_URL = os.environ.get(
    "GEMINI_MODEL_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
).strip()

GEMINI_HEADER: Dict[str, str] = {}
if "GEMINI_API_KEY" in os.environ:
  GEMINI_HEADER = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.environ.get('GEMINI_API_KEY')}",
  }

ANTHROPIC_HEADER: Dict[str, str] = {}
if "ANTHROPIC_API_KEY" in os.environ:
  ANTHROPIC_HEADER = {
      "Content-Type": "application/json",
      "Anthropic-Version": "2023-06-01",
      "X-Api-Key": os.environ["ANTHROPIC_API_KEY"],
  }

CLAUDE_MODELS = ["claude-3-5-sonnet-20241022"]
# ----------------------------
# Pricing tables (USD per 1M tokens)
# ----------------------------
# Notes:
# - OpenAI: thinking tokens are included in output token billing.
# - OpenAI may provide cached input token counts in usage.prompt_tokens_details.cached_tokens.
# - Gemini pricing page states output price includes thinking tokens.
#
# You can extend these dicts as needed. Unknown models will yield cost_usd=None.

OPENAI_PRICES_PER_1M: Dict[str, Dict[str, float]] = {
    # GPT-5 
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},

    # GPT-5.2
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "cached_input": 2.10, "output": 168.00},

    "gpt-4o": {"input": 5.00, "cached_input": 0.50, "output": 15.00},
    "o1-preview": {"input": 15.00, "cached_input": 1.50, "output": 60.00},
    "o1": {"input": 15.00, "cached_input": 1.50, "output": 60.00},
}

GEMINI_PRICES_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
"gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
}


# ----------------------------
# Qwen vLLM base_url builder
# ----------------------------

def _build_qwen_base_url() -> str:
  """
  Build a stable OpenAI-compatible base_url for vLLM Qwen server.

  Priority:
    1) QWEN_BASE_URL (if provided, should include scheme and may include /v1)
    2) QWEN_IP + QWEN_PORT

  Returns a string ending with "/v1".
  """
  explicit = os.environ.get("QWEN_BASE_URL", "").strip()
  if explicit:
    base = explicit.rstrip("/")
    if not base.endswith("/v1"):
      base = base + "/v1"
    return base

  qwen_ip = os.environ.get("QWEN_IP", "127.0.0.1").strip()
  qwen_port = os.environ.get("QWEN_PORT", "8000").strip()

  if "://" not in qwen_ip:
    qwen_ip = "http://" + qwen_ip

  parsed = urlparse(qwen_ip)
  base = qwen_ip.rstrip("/")

  # If ip already has a port, do not append another.
  if parsed.port is None and qwen_port:
    base = f"{base}:{qwen_port}"

  if not base.endswith("/v1"):
    base = base + "/v1"
  return base


QWEN_BASE_URL = _build_qwen_base_url()

# Sync OpenAI client for Qwen (does not cause event-loop issues)
client = OpenAI(
    base_url=QWEN_BASE_URL,
    api_key="EMPTY",
)

# Tokenizer for Qwen chat template
# You can override this to a local path to avoid HF downloads:
#   export QWEN_TOKENIZER_PATH=/path/to/local/tokenizer
_QWEN_TOKENIZER_PATH = os.environ.get(
    "QWEN_TOKENIZER_PATH",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
)
tokenizer = transformers.AutoTokenizer.from_pretrained(_QWEN_TOKENIZER_PATH)


# ----------------------------
# Cache
# ----------------------------

def load_cache_file(cache_file: str) -> Dict[str, Any]:
  """
  Cache schema (jsonl lines):
    - legacy: {"prompt": <json_str>, "completion": <str or dict>}
    - qwen legacy: {"prompt": <json_str>, "completion": <str>, "num_thinking_tokens": <int>, "cot": <str>}
    - unified (this file): {"prompt": <json_str>, "completion": <str>, "num_thinking_tokens": <int>, "cot": <str>, "cost_usd": <float>, "usage": {...}}
  """
  cache: Dict[str, Any] = {}
  if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
      for line in f:
        try:
          obj = json.loads(line)
        except Exception:
          continue
        if "prompt" not in obj:
          continue
        jp = obj["prompt"]

        # Unified entry
        if isinstance(obj, dict) and "completion" in obj and ("num_thinking_tokens" in obj or "cot" in obj or "cost_usd" in obj):
          cache[jp] = {
              "completion": obj.get("completion", ""),
              "num_thinking_tokens": obj.get("num_thinking_tokens", None),
              "cot": obj.get("cot", None),
              "cost_usd": obj.get("cost_usd", None),
              "usage": obj.get("usage", {}) if isinstance(obj.get("usage", {}), dict) else {},
              "model": obj.get("model", None),
          }
          continue

        # Older qwen style
        if "num_thinking_tokens" in obj and "cot" in obj and "completion" in obj:
          cache[jp] = (obj.get("completion", ""), obj.get("num_thinking_tokens", 0), obj.get("cot", ""))
          continue

        # Legacy: {"prompt": ..., "completion": ...}
        if "completion" in obj:
          cache[jp] = obj["completion"]
  return cache


def jsonify_prompt(prompt) -> str:
  return json.dumps(prompt, ensure_ascii=False)


# ----------------------------
# HTTP helpers
# ----------------------------

@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def openai_like_request(model_url: str, data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
  response = requests.post(model_url, headers=headers, json=data)
  try:
    response_json = response.json()
    assert "choices" in response_json
  except Exception as e:
    print(getattr(response, "text", response))
    raise e
  return response_json


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def claude_request(model_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
  response = requests.post(model_url, headers=ANTHROPIC_HEADER, json=data)
  try:
    response_json = response.json()
    assert "content" in response_json
  except Exception as e:
    print(getattr(response, "text", response))
    raise e
  return response_json


# ----------------------------
# Gemma adapter (kept)
# ----------------------------

def process_gemma_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
  processed_messages: List[Dict[str, str]] = []
  last_role: Optional[str] = None

  for message in messages:
    current_role = message["role"]
    if current_role == "system":
      current_role = "user"
      message = {"role": "user", "content": message["content"]}

    if current_role == last_role:
      processed_messages[-1]["content"] += "\n\n" + message["content"]
    else:
      processed_messages.append(message)
      last_role = current_role

  final_messages: List[Dict[str, str]] = []
  for i, message in enumerate(processed_messages):
    if i == 0 and message["role"] != "user":
      final_messages.append({"role": "user", "content": "Hello"})

    if i > 0 and message["role"] == final_messages[-1]["role"]:
      if message["role"] == "user":
        final_messages.append({"role": "assistant", "content": "I understand."})
      else:
        final_messages.append({"role": "user", "content": "Please continue."})

    final_messages.append(message)

  return final_messages


# ----------------------------
# Cost and token parsing
# ----------------------------

def _lower(s: Any) -> str:
  try:
    return str(s).lower()
  except Exception:
    return ""


def _pick_openai_price(model_name: str) -> Optional[Dict[str, float]]:
  mn = _lower(model_name)
  # prefer longest key match
  matches: List[Tuple[int, str]] = []
  for k in OPENAI_PRICES_PER_1M:
    if _lower(k) in mn:
      matches.append((len(k), k))
  if not matches:
    return None
  matches.sort(reverse=True)
  return OPENAI_PRICES_PER_1M[matches[0][1]]


def _pick_gemini_price(model_name: str) -> Optional[Dict[str, float]]:
  mn = _lower(model_name)
  matches: List[Tuple[int, str]] = []
  for k in GEMINI_PRICES_PER_1M:
    if _lower(k) in mn:
      matches.append((len(k), k))
  if not matches:
    return None
  matches.sort(reverse=True)
  return GEMINI_PRICES_PER_1M[matches[0][1]]


def _extract_usage_openai_like(resp: Dict[str, Any]) -> Dict[str, Any]:
  usage = resp.get("usage", {}) or {}
  ptd = usage.get("prompt_tokens_details", {}) or {}
  cached_tokens = ptd.get("cached_tokens", 0) or 0

  ctd = usage.get("completion_tokens_details", {}) or {}
  otd = usage.get("output_tokens_details", {}) or {}

  reasoning_tokens = None
  if "reasoning_tokens" in ctd and ctd["reasoning_tokens"] is not None:
    reasoning_tokens = ctd["reasoning_tokens"]
  elif "reasoning_tokens" in otd and otd["reasoning_tokens"] is not None:
    reasoning_tokens = otd["reasoning_tokens"]

  return {
      "prompt_tokens": usage.get("prompt_tokens", None),
      "completion_tokens": usage.get("completion_tokens", None),
      "total_tokens": usage.get("total_tokens", None),
      "cached_prompt_tokens": cached_tokens,
      "reasoning_tokens": reasoning_tokens,
  }


def _compute_cost_openai_like(model_name: str, usage: Dict[str, Any], provider: str) -> Optional[float]:
  if not isinstance(usage, dict):
    return None
  pt = usage.get("prompt_tokens", None)
  ct = usage.get("completion_tokens", None)
  if pt is None or ct is None:
    return None

  try:
    pt_i = int(pt)
    ct_i = int(ct)
  except Exception:
    return None

  if provider == "openai":
    price = _pick_openai_price(model_name)
    if price is None:
      return None
    cached = usage.get("cached_prompt_tokens", 0) or 0
    try:
      cached_i = int(cached)
    except Exception:
      cached_i = 0
    uncached_i = max(0, pt_i - cached_i)

    in_cost = (uncached_i * float(price["input"])) / 1_000_000.0
    cached_price = price.get("cached_input", None)
    if cached_price is not None and cached_i > 0:
      in_cost += (cached_i * float(cached_price)) / 1_000_000.0
    out_cost = (ct_i * float(price["output"])) / 1_000_000.0
    return float(in_cost + out_cost)

  if provider == "gemini":
    price = _pick_gemini_price(model_name)
    if price is None:
      return None
    in_cost = (pt_i * float(price["input"])) / 1_000_000.0
    out_cost = (ct_i * float(price["output"])) / 1_000_000.0
    return float(in_cost + out_cost)

  return None


def _extract_text_and_thought_openai_like(resp: Dict[str, Any]) -> Tuple[str, str]:
  """
  Return (final_output, cot_like_text).

  For OpenAI: cot is not available, so returns "".
  For Gemini OpenAI compatibility: content may include thought-like parts, we best-effort capture them.
  """
  try:
    msg = resp["choices"][0]["message"]
  except Exception:
    return (str(resp), "")

  content = msg.get("content", "")
  if isinstance(content, str):
    return (content.strip(), "")

  # Some compat layers may return list parts
  final_parts: List[str] = []
  thought_parts: List[str] = []

  if isinstance(content, list):
    for part in content:
      if isinstance(part, str):
        final_parts.append(part)
        continue
      if not isinstance(part, dict):
        continue
      txt = part.get("text") or part.get("content") or ""
      if part.get("thought") is True or "thoughtSignature" in part:
        if txt:
          thought_parts.append(str(txt))
      else:
        if txt:
          final_parts.append(str(txt))

  return ("\n".join(final_parts).strip(), "\n".join(thought_parts).strip())


# ----------------------------
# Qwen async generation (kept)
# ----------------------------

def _cap_concurrency(max_concurrent: int, batch_size: int) -> int:
  try:
    mc = int(max_concurrent)
  except Exception:
    mc = 8
  mc = max(1, mc)
  if batch_size > 0:
    mc = min(mc, batch_size)
  mc = min(mc, 64)
  return mc


async def qwen_async_batch_generate(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    max_concurrent: int = 32,
) -> List[Tuple[str, int, str]]:
  """
  Returns List[Tuple[final_output, num_thinking_tokens, cot]]

  Key design: do NOT use a module-level AsyncOpenAI client.
  """
  batch_size = len(batch_messages)
  max_concurrent = _cap_concurrency(max_concurrent, batch_size)
  semaphore = asyncio.Semaphore(max_concurrent)

  timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=300.0)
  limits = httpx.Limits(
      max_connections=max_concurrent,
      max_keepalive_connections=max_concurrent,
      keepalive_expiry=30.0,
  )

  retryable = (APIConnectionError, APITimeoutError, RateLimitError, httpx.RequestError, asyncio.TimeoutError)

  async def _call_one(async_client: AsyncOpenAI, raw_prompt_text: str):
    async for attempt in tenacity.AsyncRetrying(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=tenacity.retry_if_exception_type(retryable),
        reraise=True,
    ):
      with attempt:
        return await async_client.completions.create(
            model=model_name,
            prompt=raw_prompt_text,
            logprobs=2,
            echo=False,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
        )

  async with httpx.AsyncClient(timeout=timeout, limits=limits) as http_client:
    async_client = AsyncOpenAI(
        base_url=QWEN_BASE_URL,
        api_key="EMPTY",
        http_client=http_client,
        max_retries=0,
    )

    async def generate_one(idx: int, messages: List[Dict[str, str]]):
      async with semaphore:
        raw_prompt_text = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_reasoning=True,
            add_special_tokens=True,
        )

        resp = await _call_one(async_client, raw_prompt_text)

        choice = resp.choices[0]
        response_text = getattr(choice, "text", "") or ""

        num_thinking_tokens = 0
        cot = ""
        final_output = response_text

        logprobs = getattr(choice, "logprobs", None)
        tokens = getattr(logprobs, "tokens", None) if logprobs else None
        if tokens and "</think>" in tokens:
          try:
            num_thinking_tokens = int(tokens.index("</think>"))
          except Exception:
            num_thinking_tokens = 0

          if "</think>" in response_text:
            cot, final_output = response_text.split("</think>", 1)

        return idx, (final_output.strip(), int(num_thinking_tokens), cot.strip())

    tasks = [generate_one(i, msg) for i, msg in enumerate(batch_messages)]
    results_with_idx = await asyncio.gather(*tasks)

  results_with_idx.sort(key=lambda x: x[0])
  return [result for _, result in results_with_idx]


def _run_async_safely(coro):
  try:
    _ = asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(coro)

  raise RuntimeError(
      "model_call_wrapper was called from within a running event loop. "
      "Refactor the caller to await qwen_async_batch_generate directly."
  )


# ----------------------------
# Model call wrapper
# ----------------------------

def _strip_gpt5_incompatible_fields(model_name: str, gen: Dict[str, Any]) -> Dict[str, Any]:
  """
  Some GPT-5 endpoints can reject unsupported sampling fields.
  We keep this conservative.
  """
  if not isinstance(gen, dict):
    return {}
  out = dict(gen)
  mn = _lower(model_name)
  if "gpt-5" in mn:
    for k in ["temperature", "top_p", "top_k", "logprobs", "echo", "frequency_penalty", "presence_penalty"]:
      out.pop(k, None)
    # Some stacks prefer max_completion_tokens for reasoning models.
    if "max_tokens" in out and "max_completion_tokens" not in out:
      out["max_completion_tokens"] = out.pop("max_tokens")
  return out


def _is_openai_chat_model_name(model_name: str) -> bool:
  mn = _lower(model_name)
  return mn.startswith("gpt-") or mn.startswith("o1") or mn.startswith("o3")


def model_call_wrapper(
    model_name: str,
    model_url: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    parallel_model_calls: bool,
) -> List[Any]:
  if not batch_messages:
    return []

  def get_batch_responses(get_response):
    if not parallel_model_calls or len(batch_messages) <= 1:
      return [get_response(m) for m in batch_messages]
    with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
      return list(executor.map(get_response, batch_messages))

  # OpenAI chat models (GPT, o1, etc)
  if _is_openai_chat_model_name(model_name):
    def get_response(messages):
      gen = _strip_gpt5_incompatible_fields(model_name, generation_config)
      data = {"model": model_name, "messages": messages, **gen}
      url = model_url if (isinstance(model_url, str) and model_url.startswith("http")) else OPENAI_CHAT_COMPLETIONS_URL
      return openai_like_request(url, data, headers=OPENAI_HEADER)
    return get_batch_responses(get_response)

  # Gemini via OpenAI compatibility endpoint (so we get usage back)
  if "gemini" in _lower(model_name):
    if not GEMINI_HEADER:
      raise ValueError("GEMINI_API_KEY is not set, cannot call Gemini OpenAI compatibility endpoint.")
    def get_response(messages):
      data = {"model": model_name, "messages": messages, **generation_config}
      url = model_url if (isinstance(model_url, str) and model_url.startswith("http")) else GEMINI_MODEL_URL
      return openai_like_request(url, data, headers=GEMINI_HEADER)
    return get_batch_responses(get_response)

  # Gemma (kept)
  if "gemma" in _lower(model_name):
    def get_response(messages):
      final_messages = process_gemma_messages(messages)
      data = {
          "model": model_name,
          "messages": final_messages,
          "temperature": 0.0,
          "max_tokens": 512,
      }
      response = requests.post(model_url, json=data)
      try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()
      except Exception as e:
        print(getattr(response, "text", response))
        raise e
    return get_batch_responses(get_response)

  # Qwen vLLM
  if "qwen" in _lower(model_name):
    return _run_async_safely(qwen_async_batch_generate(model_name, batch_messages))

  # Claude (kept)
  if "claude" in _lower(model_name):
    def get_response(messages):
      converted = []
      for message in messages:
        role = message["role"]
        if role == "system":
          role = "user"
        converted.append({"role": role, "content": message.get("content", "")})
      data = {"model": model_name, "messages": converted, **generation_config}
      return claude_request(model_url, data)
    return get_batch_responses(get_response)

  raise ValueError(f"Unsupported model_name: {model_name}")


# ----------------------------
# Normalization to unified (text, thinking, cot, cost)
# ----------------------------

def _normalize_one(model_name: str, raw: Any) -> Tuple[str, Optional[int], Optional[str], Optional[float], Dict[str, Any]]:
  """
  Returns:
    completion: str
    num_thinking_tokens: Optional[int]
    cot: Optional[str]
    cost_usd: Optional[float]
    usage: Dict[str, Any]
  """
  mn = _lower(model_name)

  # Qwen already returns (final_output, num_thinking_tokens, cot)
  if "qwen" in mn and isinstance(raw, tuple) and len(raw) == 3:
    completion, ntok, cot = raw
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": int(ntok) if ntok is not None else None,
    }
    return str(completion), int(ntok) if ntok is not None else None, str(cot), None, usage

  # OpenAI-like dict (OpenAI GPT or Gemini compat)
  if isinstance(raw, dict) and "choices" in raw:
    completion, thought = _extract_text_and_thought_openai_like(raw)
    usage = _extract_usage_openai_like(raw)
    # thinking tokens: prefer reasoning_tokens if present, else 0
    ntok_val = usage.get("reasoning_tokens", None)
    try:
      ntok = int(ntok_val) if ntok_val is not None else 0
    except Exception:
      ntok = 0

    raw_model = raw.get("model", model_name) if isinstance(raw.get("model", None), str) else model_name
    provider = "gemini" if "gemini" in _lower(raw_model) or "gemini" in mn else "openai"
    cost_usd = _compute_cost_openai_like(raw_model, usage, provider=provider)

    cot = thought if thought else ""
    return completion, ntok, cot, cost_usd, usage

  # Claude format
  if isinstance(raw, dict) and "content" in raw and isinstance(raw.get("content"), list):
    try:
      completion = raw["content"][0]["text"]
    except Exception:
      completion = str(raw)
    return completion, None, None, None, {}

  # Fallback string
  return str(raw), None, None, None, {}


def _normalize_outputs(model_name: str, raw_responses: List[Any]):
  texts: List[str] = []
  thinking: List[Optional[int]] = []
  cots: List[Optional[str]] = []
  costs: List[Optional[float]] = []

  for r in raw_responses:
    t, nt, c, usd, _usage = _normalize_one(model_name, r)
    texts.append(t)
    thinking.append(nt)
    cots.append(c)
    costs.append(usd)

  return texts, thinking, cots, costs


# ----------------------------
# cached_generate
# ----------------------------

def cached_generate(
    batch_prompts: List[List[Dict[str, str]]],
    model_name: str,
    model_url: str,
    cache: Optional[Dict[str, Any]],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    parallel_model_calls: bool,
):
  """
  Returns:
    batch_text_outputs: List[str]
    all_num_thinking_tokens: List[Optional[int]]
    all_cots: List[Optional[str]]
    all_costs_usd: List[Optional[float]]
  """
  # o1 requires no system role
  if model_name.startswith("o1"):
    for prompt in batch_prompts:
      for t, turn in enumerate(prompt):
        if turn["role"] == "system":
          prompt[t]["role"] = "user"
    generation_config = {}

  # No cache mode
  if cache is None:
    raw = model_call_wrapper(
        model_name=model_name,
        model_url=model_url,
        batch_messages=batch_prompts,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )
    return _normalize_outputs(model_name, raw)

  # Decide which prompts are new
  new_batch_prompts: List[List[Dict[str, str]]] = []
  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    if jp not in cache:
      new_batch_prompts.append(prompt)
      continue
    # old cache entries might be raw strings; we still accept but do not requery
    # unless it is an OpenAI-like dict missing "choices"
    entry = cache[jp]
    if isinstance(entry, dict) and "choices" in entry:
      # already a raw OpenAI-like response, ok
      continue

  # Query new prompts
  if new_batch_prompts:
    raw_responses = model_call_wrapper(
        model_name=model_name,
        model_url=model_url,
        batch_messages=new_batch_prompts,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )

    # write into cache as unified dict
    for prompt, raw in zip(new_batch_prompts, raw_responses):
      jp = jsonify_prompt(prompt)
      completion, ntok, cot, cost_usd, usage = _normalize_one(model_name, raw)

      unified = {
          "completion": completion,
          "num_thinking_tokens": ntok,
          "cot": cot,
          "cost_usd": cost_usd,
          "usage": usage,
          "model": raw.get("model", model_name) if isinstance(raw, dict) else model_name,
      }
      cache[jp] = unified

      if cache_file:
        parent = os.path.dirname(cache_file)
        if parent:
          os.makedirs(parent, exist_ok=True)
        with open(cache_file, "a") as f:
          f.write(json.dumps({"prompt": jp, **unified}, ensure_ascii=False) + "\n")

  # Build outputs in original order
  batch_text_outputs: List[str] = []
  all_num_thinking_tokens: List[Optional[int]] = []
  all_cots: List[Optional[str]] = []
  all_costs_usd: List[Optional[float]] = []

  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    entry = cache.get(jp)

    # Unified dict
    if isinstance(entry, dict) and "completion" in entry:
      batch_text_outputs.append(str(entry.get("completion", "")))
      all_num_thinking_tokens.append(entry.get("num_thinking_tokens", None))
      all_cots.append(entry.get("cot", None))
      all_costs_usd.append(entry.get("cost_usd", None))
      continue

    # Older qwen tuple
    if isinstance(entry, tuple) and len(entry) == 3:
      completion, ntok, cot = entry
      batch_text_outputs.append(str(completion))
      try:
        all_num_thinking_tokens.append(int(ntok))
      except Exception:
        all_num_thinking_tokens.append(None)
      all_cots.append(str(cot))
      all_costs_usd.append(None)
      continue

    # Raw OpenAI-like dict cached
    if isinstance(entry, dict) and "choices" in entry:
      completion, ntok, cot, cost_usd, _usage = _normalize_one(model_name, entry)
      batch_text_outputs.append(completion)
      all_num_thinking_tokens.append(ntok)
      all_cots.append(cot)
      all_costs_usd.append(cost_usd)
      continue

    # Fallback string
    batch_text_outputs.append(str(entry) if entry is not None else "")
    all_num_thinking_tokens.append(None)
    all_cots.append(None)
    all_costs_usd.append(None)

  return batch_text_outputs, all_num_thinking_tokens, all_cots, all_costs_usd