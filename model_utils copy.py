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

from concurrent import futures
import json
import os
from typing import Dict, List, Tuple, Any, Optional
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

if "GOOGLE_API_KEY" in os.environ:
  import google.generativeai as genai
  genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

OPENAI_HEADER = {}
if "OPENAI_API_KEY" in os.environ:
  OPENAI_HEADER = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
      "OpenAI-Organization": os.environ.get("OPENAI_ORGANIZATION"),
      "OpenAI-Project": os.environ.get("OPENAI_PROJECT"),
  }

ANTHROPIC_HEADER = {}
if "ANTHROPIC_API_KEY" in os.environ:
  ANTHROPIC_HEADER = {
      "Content-Type": "application/json",
      "Anthropic-Version": "2023-06-01",
      "X-Api-Key": os.environ["ANTHROPIC_API_KEY"],
  }

GPT_COSTS = {
    "gpt-4o": {
        "prompt_tokens": 5 / 1000000,
        "completion_tokens": 15 / 1000000,
    },
    "o1-preview": {
        "prompt_tokens": 15 / 1000000,
        "completion_tokens": 60 / 1000000,
    },
    "o1": {
        "prompt_tokens": 15 / 1000000,
        "completion_tokens": 60 / 1000000,
    },
}

CLAUDE_MODELS = ["claude-3-5-sonnet-20241022"]

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

# Keep a sync client if you ever use it elsewhere. It does not cause the event-loop issue.
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

def load_cache_file(cache_file: str) -> Dict[str, Any]:
  """
  Cache schema (jsonl lines):
    - legacy: {"prompt": <json_str>, "completion": <str or dict>}
    - qwen:   {"prompt": <json_str>, "completion": <str>, "num_thinking_tokens": <int>, "cot": <str>}
  """
  cache: Dict[str, Any] = {}
  if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
      for line in f:
        line = json.loads(line)
        if "num_thinking_tokens" in line and "cot" in line:
          cache[line["prompt"]] = (
              line["completion"],
              line["num_thinking_tokens"],
              line["cot"],
          )
        else:
          cache[line["prompt"]] = line["completion"]
  return cache

def jsonify_prompt(prompt) -> str:
  return json.dumps(prompt)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def openai_request(model_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
  response = requests.post(model_url, headers=OPENAI_HEADER, json=data)
  try:
    response = response.json()
    assert "choices" in response
  except Exception as e:
    print(response)
    raise e
  return response

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

@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def claude_request(model_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
  response = requests.post(model_url, headers=ANTHROPIC_HEADER, json=data)
  try:
    response = response.json()
    assert "content" in response
  except Exception as e:
    print(response)
    raise e
  return response

def _cap_concurrency(max_concurrent: int, batch_size: int) -> int:
  try:
    mc = int(max_concurrent)
  except Exception:
    mc = 8
  mc = max(1, mc)
  if batch_size > 0:
    mc = min(mc, batch_size)
  # Avoid overwhelming a local vLLM server by default.
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
  We create and close the async client within the same event loop,
  preventing "Event loop is closed" during connection pool cleanup.
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
        max_retries=0,  # handled by tenacity above
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

        # best-effort thinking token split
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
  """
  Run an async coroutine from sync code.

  If already inside a running event loop in the same thread, asyncio.run cannot be used.
  In this codebase model_call_wrapper is expected to be called from sync code.
  """
  try:
    _ = asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(coro)

  raise RuntimeError(
      "model_call_wrapper was called from within a running event loop. "
      "Refactor the caller to await qwen_async_batch_generate directly."
  )

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

  if model_name in GPT_COSTS:
    def get_response(messages):
      data = {"model": model_name, "messages": messages, **generation_config}
      return openai_request(model_url, data)
    return get_batch_responses(get_response)

  if "gemini" in model_name.lower():
    def get_response(messages):
      model = genai.GenerativeModel(model_url)
      converted = []
      for message in messages:
        role = message["role"]
        if role == "system":
          role = "user"
        converted.append({"role": role, "parts": message.get("content", "")})
      chat = model.start_chat(history=converted[:-1])
      return chat.send_message(converted[-1]).text
    return get_batch_responses(get_response)

  if "gemma" in model_name.lower():
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
        print(response.text)
        raise e
    return get_batch_responses(get_response)

  if "qwen" in model_name.lower():
    # returns List[Tuple[final_output, num_thinking_tokens, cot]]
    return _run_async_safely(qwen_async_batch_generate(model_name, batch_messages))

  if model_name in CLAUDE_MODELS:
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

  # critical: do not silently return None
  raise ValueError(f"Unsupported model_name: {model_name}")

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
  """
  # o1 requires no system role
  if model_name.startswith("o1"):
    for prompt in batch_prompts:
      for t, turn in enumerate(prompt):
        if turn["role"] == "system":
          prompt[t]["role"] = "user"
    generation_config = {}

  # No cache mode: still keep return shape consistent
  if cache is None:
    raw = model_call_wrapper(
        model_name=model_name,
        model_url=model_url,
        batch_messages=batch_prompts,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )
    text_out, thinking_tokens, cots = _normalize_outputs_from_raw(model_name, raw)
    return text_out, thinking_tokens, cots

  new_batch_prompts: List[List[Dict[str, str]]] = []
  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    if jp not in cache:
      new_batch_prompts.append(prompt)
    elif model_name in GPT_COSTS and isinstance(cache[jp], dict) and "choices" not in cache[jp]:
      new_batch_prompts.append(prompt)

  if new_batch_prompts:
    raw_responses = model_call_wrapper(
        model_name=model_name,
        model_url=model_url,
        batch_messages=new_batch_prompts,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )

    # write into cache
    for prompt, raw in zip(new_batch_prompts, raw_responses):
      jp = jsonify_prompt(prompt)
      cache[jp] = raw

      if cache_file:
        parent = os.path.dirname(cache_file)
        if parent:
          os.makedirs(parent, exist_ok=True)

        with open(cache_file, "a") as f:
          if isinstance(raw, tuple) and len(raw) == 3:
            completion, num_thinking_tokens, cot = raw
            f.write(
                json.dumps({
                    "prompt": jp,
                    "completion": completion,
                    "num_thinking_tokens": num_thinking_tokens,
                    "cot": cot,
                }) + "\n"
            )
          else:
            f.write(
                json.dumps({
                    "prompt": jp,
                    "completion": raw,
                }) + "\n"
            )

    assert len(raw_responses) == len(new_batch_prompts)

  batch_text_outputs: List[str] = []
  all_num_thinking_tokens: List[Optional[int]] = []
  all_cots: List[Optional[str]] = []

  for prompt in batch_prompts:
    jp = jsonify_prompt(prompt)
    entry = cache[jp]

    if model_name in GPT_COSTS:
      text_output = entry["choices"][0]["message"]["content"]
      batch_text_outputs.append(text_output)
      all_num_thinking_tokens.append(None)
      all_cots.append(None)
    elif model_name in CLAUDE_MODELS:
      text_output = entry["content"][0]["text"]
      batch_text_outputs.append(text_output)
      all_num_thinking_tokens.append(None)
      all_cots.append(None)
    else:
      if isinstance(entry, tuple) and len(entry) == 3:
        completion, num_thinking_tokens, cot = entry
        batch_text_outputs.append(completion)
        all_num_thinking_tokens.append(int(num_thinking_tokens))
        all_cots.append(cot)
      else:
        batch_text_outputs.append(str(entry))
        all_num_thinking_tokens.append(None)
        all_cots.append(None)

  return batch_text_outputs, all_num_thinking_tokens, all_cots

def _normalize_outputs_from_raw(model_name: str, raw_responses: List[Any]):
  """
  Normalize raw model_call_wrapper outputs to:
    text_outputs: List[str]
    thinking_tokens: List[Optional[int]]
    cots: List[Optional[str]]
  """
  text_outputs: List[str] = []
  thinking_tokens: List[Optional[int]] = []
  cots: List[Optional[str]] = []

  if model_name in GPT_COSTS:
    for r in raw_responses:
      text_outputs.append(r["choices"][0]["message"]["content"])
      thinking_tokens.append(None)
      cots.append(None)
    return text_outputs, thinking_tokens, cots

  if model_name in CLAUDE_MODELS:
    for r in raw_responses:
      text_outputs.append(r["content"][0]["text"])
      thinking_tokens.append(None)
      cots.append(None)
    return text_outputs, thinking_tokens, cots

  if "qwen" in model_name.lower():
    for r in raw_responses:
      if isinstance(r, tuple) and len(r) == 3:
        final_out, ntok, cot = r
        text_outputs.append(final_out)
        thinking_tokens.append(int(ntok))
        cots.append(cot)
      else:
        text_outputs.append(str(r))
        thinking_tokens.append(None)
        cots.append(None)
    return text_outputs, thinking_tokens, cots

  for r in raw_responses:
    text_outputs.append(str(r))
    thinking_tokens.append(None)
    cots.append(None)
  return text_outputs, thinking_tokens, cots