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

"""Utility functions for calling models."""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI, AsyncAzureOpenAI
import httpx
import tenacity
from tenacity import retry
import transformers

import google.generativeai as genai

wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt


@dataclass
class GenerationResult:
  """Result from a single generation call."""
  text: str
  num_thinking_tokens: int = 0
  cot: str = ""
  cost_usd: float = 0.0


@dataclass
class LocalModelConfig:
  """Configuration for a local model."""
  tokenizer_name: str
  base_url: str = "http://0.0.0.0:8011/v1"
  enable_reasoning: bool = True
  thinking_end_token: str = "</think>"


# Local models hosted via vLLM
LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = {
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": LocalModelConfig(
        tokenizer_name="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        enable_reasoning=True,
        thinking_end_token="</think>",
    ),
    "Qwen/Qwen3-4B-A3B-Thinking-2507-FP8": LocalModelConfig(
        tokenizer_name="Qwen/Qwen3-4B-Thinking-2507-FP8",
        enable_reasoning=True,
        thinking_end_token="</think>",
    ),
    "openai/gpt-oss-20b": LocalModelConfig(
        tokenizer_name="openai/gpt-oss-20b",
        enable_reasoning=True,
        thinking_end_token="",  # FIXME
    ),
    "mistralai/Magistral-Small-2507": LocalModelConfig(
        tokenizer_name="mistralai/Magistral-Small-2507",
        enable_reasoning=True,
        thinking_end_token="[/THINK]",
    ),
}

if "GOOGLE_API_KEY" in os.environ:
  genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Remote GPT models via Azure OpenAI
GPT_COSTS = {
    "gpt-5": {
        "prompt_tokens": 1.25 / 1000000,
        "completion_tokens": 10 / 1000000,
    },
    "gpt-5-mini": {
        "prompt_tokens": 0.25 / 1000000,
        "completion_tokens": 2 / 1000000,
    },
    "gpt-4.1": {
        "prompt_tokens": 2 / 1000000,
        "completion_tokens": 8 / 1000000,
    },
    "gpt-4.1-mini": {
        "prompt_tokens": 0.4 / 1000000,
        "completion_tokens": 1.60 / 1000000,
    },
}

GEMINI_TEXT_COSTS = {
    # gemini-3-pro-preview (text + thinking)
    "gemini-3-pro-preview": {
        "in": 2.00,
        "out": 12.00,
    },
    # gemini-3-flash-preview (text)
    "gemini-3-flash-preview": {
        "in": 0.50,
        "out": 2.00,
    },
}

def _normalize_gemini_model_name(model: str) -> str:
  # Handle variants like "gemini-2.5-flash-latest" by prefix match
  return model.strip()

def _gemini_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
  m = _normalize_gemini_model_name(model)

  # find best prefix match
  key = None
  for k in GEMINI_TEXT_COSTS.keys():
    if m == k or m.startswith(k):
      key = k
      break
  if key is None:
    return 0.0

  spec = GEMINI_TEXT_COSTS[key]
  pt = float(prompt_tokens or 0)
  ct = float(completion_tokens or 0)

  if "threshold" in spec:
    thr = int(spec["threshold"])
    if prompt_tokens <= thr:
      in_rate = float(spec["in_le_200k"])
      out_rate = float(spec["out_le_200k"])
    else:
      in_rate = float(spec["in_gt_200k"])
      out_rate = float(spec["out_gt_200k"])
  else:
    in_rate = float(spec["in"])
    out_rate = float(spec["out"])

  return pt * in_rate / 1_000_000.0 + ct * out_rate / 1_000_000.0

# Claude models
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

# Gemini models (using google-generativeai here)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]

_azure_openai_client: Optional[AsyncAzureOpenAI] = None
_anthropic_client: Optional[httpx.AsyncClient] = None
_local_client: Optional[AsyncOpenAI] = None

_tokenizer: Dict[str, Any] = {}


def get_tokenizer(tokenizer_name: str):
  """Get or create a tokenizer."""
  global _tokenizer
  _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  return _tokenizer


def get_local_model_config(model_name: str) -> Optional[LocalModelConfig]:
  """Get config for a local model, returns None if not found."""
  for key, config in LOCAL_MODEL_CONFIGS.items():
    if key in model_name:
      return config
  return None


def register_local_model(
    model_key: str,
    tokenizer_name: str,
    base_url: str = "http://0.0.0.0:8011/v1",
    enable_reasoning: bool = True,
    thinking_end_token: str = "</think>",
):
  """Register a new local model configuration."""
  LOCAL_MODEL_CONFIGS[model_key] = LocalModelConfig(
      tokenizer_name=tokenizer_name,
      base_url=base_url,
      enable_reasoning=enable_reasoning,
      thinking_end_token=thinking_end_token,
  )


def get_azure_openai_client() -> AsyncAzureOpenAI:
  """Get or create the Azure OpenAI async client."""
  global _azure_openai_client
  if _azure_openai_client is None:
    _azure_openai_client = AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-10-21",
    )
    
  return _azure_openai_client


def get_anthropic_client() -> httpx.AsyncClient:
  """Get or create the Anthropic async client."""
  global _anthropic_client
  if _anthropic_client is None:
    _anthropic_client = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        headers={
            "Content-Type": "application/json",
            "Anthropic-Version": "2023-06-01",
            "X-Api-Key": os.environ.get("ANTHROPIC_API_KEY", ""),
        },
        timeout=120.0,
    )
  return _anthropic_client


def get_local_client(base_url: str = "http://0.0.0.0:8011/v1") -> AsyncOpenAI:
  """Get or create the local vLLM async client."""
  global _local_client
  if _local_client is None:
    _local_client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )
  return _local_client


def load_cache_file(cache_file: str) -> Dict[str, Any]:
  """Load cache from a JSONL file."""
  cache: Dict[str, Any] = {}
  if not os.path.exists(cache_file):
    return cache

  with open(cache_file, "r") as f:
    for line in f:
      entry = json.loads(line)
      prompt = entry["prompt"]

      text = entry.get("completion", entry.get("text", ""))
      cache[prompt] = {
          "text": text,
          "num_thinking_tokens": entry.get("num_thinking_tokens", 0),
          "cot": entry.get("cot", ""),
          "cost_usd": entry.get("cost_usd", 0.0),
      }
  return cache


def jsonify_prompt(prompt: List[Dict[str, str]]) -> str:
  """Convert prompt to JSON string for caching."""
  return json.dumps(prompt)


def _gpt_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
  if model not in GPT_COSTS:
    return 0.0
  p = GPT_COSTS[model]["prompt_tokens"]
  c = GPT_COSTS[model]["completion_tokens"]
  return float(prompt_tokens) * p + float(completion_tokens) * c


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def openai_chat_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> Dict[str, Any]:
  client = get_azure_openai_client()
  response = await client.chat.completions.create(
      model=model,
      messages=messages,
      **generation_config
  )

  usage = response.usage
  # Chat Completions: usage.completion_tokens_details.reasoning_tokens
  reasoning_tokens = 0
  try:
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
      reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    else:
      # 兜底：有些 SDK/网关可能把它放在顶层
      reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)
  except Exception:
    reasoning_tokens = 0

  return {
      "choices": [{"message": {"content": response.choices[0].message.content}}],
      "usage": {
          "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
          "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
          "reasoning_tokens": reasoning_tokens,
      }
  }
# async def openai_chat_request(
#     model: str,
#     messages: List[Dict[str, str]],
#     **generation_config
# ) -> Dict[str, Any]:
#   """Async request to OpenAI Chat Completions API."""
#   client = get_azure_openai_client()
#   response = await client.chat.completions.create(
#       model=model,
#       messages=messages,
#       **generation_config
#   )
#   return {
#       "choices": [{"message": {"content": response.choices[0].message.content}}],
#       "usage": {
#           "prompt_tokens": response.usage.prompt_tokens,
#           "completion_tokens": response.usage.completion_tokens,
#       }
#   }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def claude_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> Dict[str, Any]:
  """Async request to Anthropic Claude API."""
  client = get_anthropic_client()

  system_content = None
  filtered_messages = []
  for msg in messages:
    if msg["role"] == "system":
      system_content = msg["content"]
    else:
      filtered_messages.append(msg)

  data = {
      "model": model,
      "messages": filtered_messages,
      "max_tokens": generation_config.get("max_tokens", 16384),
  }
  if system_content:
    data["system"] = system_content

  for key in ["temperature", "top_p"]:
    if key in generation_config:
      data[key] = generation_config[key]

  response = await client.post("/v1/messages", json=data)
  response.raise_for_status()
  result = response.json()

  if "content" not in result:
    raise ValueError(f"Unexpected response format: {result}")

  return result


# def _extract_gemini_usage(resp: Any) -> Tuple[int, int]:
#   """
#   Try to extract prompt/output token counts from different SDK response shapes.
#   """
#   usage = getattr(resp, "usage_metadata", None)
#   if usage is None and isinstance(resp, dict):
#     usage = resp.get("usage_metadata") or resp.get("usageMetadata")

#   pt = 0
#   ct = 0

#   # usage may be an object with attributes (prompt_token_count, candidates_token_count)
#   if usage is not None:
#     pt = getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0) or 0
#     ct = getattr(usage, "candidates_token_count", 0) or getattr(usage, "candidatesTokenCount", 0) or 0

#     # or dict-like
#     if isinstance(usage, dict):
#       pt = usage.get("prompt_token_count", usage.get("promptTokenCount", 0)) or 0
#       ct = usage.get("candidates_token_count", usage.get("candidatesTokenCount", 0)) or 0

#   return int(pt), int(ct)

def _extract_gemini_usage(resp: Any) -> Tuple[int, int, int, int]:
  """
  Returns: (prompt_tokens, candidates_tokens, thoughts_tokens, total_tokens)
  """
  usage = getattr(resp, "usage_metadata", None)
  if usage is None and isinstance(resp, dict):
    usage = resp.get("usage_metadata") or resp.get("usageMetadata")

  pt = ct = tt = tot = 0

  if usage is not None:
    # object-like
    pt = getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0) or 0
    ct = getattr(usage, "candidates_token_count", 0) or getattr(usage, "candidatesTokenCount", 0) or 0
    tt = getattr(usage, "thoughts_token_count", 0) or getattr(usage, "thoughtsTokenCount", 0) or 0
    tot = getattr(usage, "total_token_count", 0) or getattr(usage, "totalTokenCount", 0) or 0

    # dict-like
    if isinstance(usage, dict):
      pt = usage.get("prompt_token_count", usage.get("promptTokenCount", pt)) or 0
      ct = usage.get("candidates_token_count", usage.get("candidatesTokenCount", ct)) or 0
      tt = usage.get("thoughts_token_count", usage.get("thoughtsTokenCount", tt)) or 0
      tot = usage.get("total_token_count", usage.get("totalTokenCount", tot)) or 0

  pt, ct, tt, tot = int(pt), int(ct), int(tt), int(tot)

  if tt == 0 and tot > 0:
    est = tot - pt - ct
    if est > 0:
      tt = est

  return pt, ct, tt, tot


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def gemini_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> GenerationResult:
  """Async request to Google Gemini API using google-generativeai SDK."""
  gemini_messages = []
  system_instruction = None

  for msg in messages:
    if msg["role"] == "system":
      system_instruction = msg["content"]
    else:
      role = "user" if msg["role"] == "user" else "model"
      gemini_messages.append({
          "role": role,
          "parts": [{"text": msg["content"]}]
      })

  combined_messages = []
  for msg in gemini_messages:
    if combined_messages and combined_messages[-1]["role"] == msg["role"]:
      combined_messages[-1]["parts"].extend(msg["parts"])
    else:
      combined_messages.append(msg)

  model_kwargs = {}
  if system_instruction:
    model_kwargs["system_instruction"] = system_instruction

  gen_model = genai.GenerativeModel(model, **model_kwargs)
  gen_config = genai.GenerationConfig(
      temperature=generation_config.get("temperature", 0.0),
      max_output_tokens=generation_config.get("max_tokens", 16384),
  )

  loop = asyncio.get_event_loop()

  if len(combined_messages) > 1:
    chat = gen_model.start_chat(history=combined_messages[:-1])
    resp = await loop.run_in_executor(
        None,
        lambda: chat.send_message(
            combined_messages[-1]["parts"][0]["text"],
            generation_config=gen_config
        )
    )
  else:
    resp = await loop.run_in_executor(
        None,
        lambda: gen_model.generate_content(
            combined_messages[0]["parts"][0]["text"],
            generation_config=gen_config
        )
    )

  text = getattr(resp, "text", None)
  if text is None:
    text = str(resp)

  # pt, ct = _extract_gemini_usage(resp)
  # cost = _gemini_cost_usd(model, pt, ct)

  # return GenerationResult(
  #     text=text,
  #     num_thinking_tokens=0,
  #     cot="",
  #     cost_usd=cost,
  # )
  
  pt, ct, tt, tot = _extract_gemini_usage(resp)

  cost = _gemini_cost_usd(model, pt, ct + tt)

  return GenerationResult(
      text=text,
      num_thinking_tokens=tt,
      cot="",
      cost_usd=cost,
  )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def local_model_request(
    model: str,
    messages: List[Dict[str, str]],
    config: Optional[LocalModelConfig] = None,
    **generation_config
) -> GenerationResult:
  """Async request to local model via vLLM."""
  if config is None:
    config = get_local_model_config(model)
    if config is None:
      raise ValueError(f"No config found for local model: {model}")

  client = get_local_client(config.base_url)
  tokenizer = get_tokenizer(config.tokenizer_name)

  apply_kwargs = {
      "conversation": messages,
      "add_generation_prompt": True,
      "tokenize": False,
  }

  if config.enable_reasoning:
    apply_kwargs["enable_reasoning"] = True
    apply_kwargs["add_special_tokens"] = True

  try:
    raw_prompt_text = tokenizer.apply_chat_template(**apply_kwargs)
  except TypeError:
    raw_prompt_text = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )

  response = await client.completions.create(
      model=model,
      prompt=raw_prompt_text,
      logprobs=2,
      echo=False,
      temperature=generation_config.get("temperature", 0.6),
      top_p=generation_config.get("top_p", 0.95),
      max_tokens=generation_config.get("max_tokens", 16384),
  )

  choice = response.choices[0]
  response_text = choice.text

  if config.enable_reasoning and config.thinking_end_token:
    tokens = choice.logprobs.tokens if choice.logprobs else []
    if config.thinking_end_token in tokens:
      num_thinking_tokens = tokens.index(config.thinking_end_token)
      cot, final_output = response_text.split(config.thinking_end_token, 1)
    elif config.thinking_end_token in response_text:
      cot, final_output = response_text.split(config.thinking_end_token, 1)
      num_thinking_tokens = len(tokenizer.encode(cot))
    else:
      num_thinking_tokens = len(tokens) if tokens else 0
      cot = response_text
      final_output = ""
  else:
    num_thinking_tokens = 0
    cot = ""
    final_output = response_text

  return GenerationResult(
      text=final_output.strip(),
      num_thinking_tokens=num_thinking_tokens,
      cot=cot.strip(),
      cost_usd=0.0,
  )


def is_local_model(model_name: str) -> bool:
  return get_local_model_config(model_name) is not None


def is_gpt_model(model_name: str) -> bool:
  return model_name in GPT_COSTS or model_name.startswith("gpt-")


async def async_batch_generate(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    max_concurrent: int = 64,
    local_model_config: Optional[LocalModelConfig] = None,
) -> List[GenerationResult]:
  """Unified async batch generation for all model types."""
  if not batch_messages:
    return []

  semaphore = asyncio.Semaphore(max_concurrent)

  async def generate_one(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, GenerationResult]:
    async with semaphore:
      try:
        if is_local_model(model_name):
          result = await local_model_request(
              model_name, messages, config=local_model_config, **generation_config
          )
          return idx, result

        if is_gpt_model(model_name):
          resp = await openai_chat_request(model_name, messages, **generation_config)
          text = resp["choices"][0]["message"]["content"]
          usage = resp.get("usage", {}) or {}
          pt = int(usage.get("prompt_tokens", 0) or 0)
          ct = int(usage.get("completion_tokens", 0) or 0)
          cost = _gpt_cost_usd(model_name, pt, ct)
          # return idx, GenerationResult(text=text, cost_usd=cost)
          usage = resp.get("usage", {}) or {}
          pt = int(usage.get("prompt_tokens", 0) or 0)
          ct = int(usage.get("completion_tokens", 0) or 0)
          rt = int(usage.get("reasoning_tokens", 0) or 0)

          cost = _gpt_cost_usd(model_name, pt, ct)
          return idx, GenerationResult(
              text=text,
              num_thinking_tokens=rt,
              cot="",
              cost_usd=cost,
          )

        if model_name in CLAUDE_MODELS or model_name.startswith("claude"):
          resp = await claude_request(model_name, messages, **generation_config)
          text = resp["content"][0]["text"]
          return idx, GenerationResult(text=text, cost_usd=0.0)

        # if model_name in GEMINI_MODELS or "gemini" in model_name.lower():
        #   text = await gemini_request(model_name, messages, **generation_config)
        #   return idx, GenerationResult(text=text, cost_usd=0.0)
        if model_name in GEMINI_MODELS or "gemini" in model_name.lower():
          result = await gemini_request(model_name, messages, **generation_config)
          return idx, result

        raise ValueError(f"Unknown model: {model_name}")

      except Exception as e:
        print(f"Error generating response for index {idx}: {e}")
        raise

  tasks = [generate_one(i, msg) for i, msg in enumerate(batch_messages)]
  results_with_idx = await asyncio.gather(*tasks, return_exceptions=True)

  processed_results = []
  for r in results_with_idx:
    if isinstance(r, Exception):
      raise r
    processed_results.append(r)

  processed_results.sort(key=lambda x: x[0])
  return [result for _, result in processed_results]


def model_call_wrapper(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
    max_concurrent: int = 64,
) -> List[GenerationResult]:
  """Wrapper for calling various types of models."""
  return asyncio.run(async_batch_generate(
      model_name,
      batch_messages=batch_messages,
      generation_config=generation_config,
      max_concurrent=max_concurrent,
      local_model_config=local_model_config,
  ))


def cached_generate(
    batch_prompts: List[List[Dict[str, str]]],
    model_name: str,
    model_url: Optional[str] = None,
    cache: Optional[Dict[str, Any]] = None,
    cache_file: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    parallel_model_calls: bool = True,
    local_model_config: Optional[LocalModelConfig] = None,
) -> Tuple[List[str], List[int], List[str], List[float]]:
  """
  Backwards-compatible cached generate.

  gsm.py expects:
    batch_responses, think_token_num, all_cots, cost_usd = cached_generate(
        batch_prompts, model_name, model_url, cache=..., cache_file=...,
        generation_config=..., parallel_model_calls=...
    )
  """
  _ = model_url  # kept for backwards compatibility

  if generation_config is None:
    generation_config = {}

  max_concurrent = 64 if parallel_model_calls else 1

  def _result_to_cache_dict(r: GenerationResult) -> Dict[str, Any]:
    return {
        "text": r.text,
        "num_thinking_tokens": r.num_thinking_tokens,
        "cot": r.cot,
        "cost_usd": r.cost_usd,
    }

  def _cache_dict_to_outputs(v: Any) -> Tuple[str, int, str, float]:
    if isinstance(v, dict):
      return (
          str(v.get("text", "")),
          int(v.get("num_thinking_tokens", 0) or 0),
          str(v.get("cot", "")),
          float(v.get("cost_usd", 0.0) or 0.0),
      )
    if isinstance(v, tuple):
      text = str(v[0]) if len(v) > 0 else ""
      nt = int(v[1]) if len(v) > 1 else 0
      cot = str(v[2]) if len(v) > 2 else ""
      cost = float(v[3]) if len(v) > 3 else 0.0
      return text, nt, cot, cost
    return str(v), 0, "", 0.0

  if cache is None:
    results = model_call_wrapper(
        model_name,
        batch_messages=batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
        max_concurrent=max_concurrent,
    )
    batch_responses = [r.text for r in results]
    all_num_thinking_tokens = [r.num_thinking_tokens for r in results]
    all_cots = [r.cot for r in results]
    cost_usd = [r.cost_usd for r in results]
    return batch_responses, all_num_thinking_tokens, all_cots, cost_usd

  new_batch_prompts = []
  new_prompt_indices = []
  for i, prompt in enumerate(batch_prompts):
    jp = jsonify_prompt(prompt)
    if jp not in cache:
      new_batch_prompts.append(prompt)
      new_prompt_indices.append(i)

  if new_batch_prompts:
    batch_results = model_call_wrapper(
        model_name,
        batch_messages=new_batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
        max_concurrent=max_concurrent,
    )

    for prompt, result in zip(new_batch_prompts, batch_results):
      jp = jsonify_prompt(prompt)
      cache[jp] = _result_to_cache_dict(result)

      if cache_file:
        cache_entry = {
            "prompt": jp,
            "completion": result.text,
            "num_thinking_tokens": result.num_thinking_tokens,
            "cot": result.cot,
            "cost_usd": result.cost_usd,
        }
        with open(cache_file, "a") as f:
          f.write(json.dumps(cache_entry) + "\n")

  batch_responses: List[str] = []
  all_num_thinking_tokens: List[int] = []
  all_cots: List[str] = []
  cost_usd: List[float] = []

  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    text_output, num_tokens, cot, cost = _cache_dict_to_outputs(cache[jp])
    batch_responses.append(text_output)
    all_num_thinking_tokens.append(num_tokens)
    all_cots.append(cot)
    cost_usd.append(cost)

  return batch_responses, all_num_thinking_tokens, all_cots, cost_usd


async def cleanup_clients():
  """Close all async clients."""
  global _azure_openai_client, _anthropic_client, _local_client

  if _anthropic_client:
    await _anthropic_client.aclose()
    _anthropic_client = None

  if _azure_openai_client:
    await _azure_openai_client.close()
    _azure_openai_client = None

  if _local_client:
    await _local_client.close()
    _local_client = None