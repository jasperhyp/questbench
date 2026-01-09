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

wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt


@dataclass
class GenerationResult:
  """Result from a single generation call."""
  text: str
  num_thinking_tokens: int = 0
  cot: str = ""


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
    "Qwen/Qwen3-4B-Thinking-2507": LocalModelConfig(
        tokenizer_name="Qwen/Qwen3-4B-Thinking-2507",
        enable_reasoning=True,
        thinking_end_token="</think>",
    ),
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": LocalModelConfig(
        tokenizer_name="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
        enable_reasoning=True,
        thinking_end_token="</think>",
    ),
    "openai/gpt-oss-20b": LocalModelConfig(
        tokenizer_name="openai/gpt-oss-20b", 
        enable_reasoning=True,
        thinking_end_token="",  # FIXME
    ),
    "openai/gpt-oss-120b": LocalModelConfig(
        tokenizer_name="openai/gpt-oss-120b", 
        enable_reasoning=True,
        thinking_end_token="",  # FIXME
    ),
    "mistralai/Ministral-3-14B-Reasoning-2512": LocalModelConfig(
        tokenizer_name="mistralai/Ministral-3-14B-Reasoning-2512",
        enable_reasoning=True,
        thinking_end_token="[/THINK]", 
    ),
    "mistralai/Ministral-3-8B-Reasoning-2512": LocalModelConfig(
        tokenizer_name="mistralai/Ministral-3-8B-Reasoning-2512",
        enable_reasoning=True,
        thinking_end_token="[/THINK]", 
    ),
}

if "GOOGLE_API_KEY" in os.environ:
  import google.generativeai as genai
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

# Claude models
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514", 
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

# Gemini models (using google-genai SDK v1.x with unified API)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]

_azure_openai_client: Optional[AsyncAzureOpenAI] = None
_anthropic_client: Optional[httpx.AsyncClient] = None
_local_client: Optional[AsyncOpenAI] = None  # Single local client

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
  cache = {}
  if not os.path.exists(cache_file):
    return cache
    
  with open(cache_file, "r") as f:
    for line in f:
      entry = json.loads(line)
      prompt = entry["prompt"]
      
      if "num_thinking_tokens" in entry and "cot" in entry:
        text = entry.get("completion", entry.get("text", ""))
        cache[prompt] = {
            "text": text,
            "num_thinking_tokens": entry.get("num_thinking_tokens", 0),
            "cot": entry.get("cot", ""),
        }
      elif "completion" in entry:
        completion = entry["completion"]
        if isinstance(completion, dict):
          if "choices" in completion:
            text = completion["choices"][0]["message"]["content"]
          elif "content" in completion:
            text = completion["content"][0]["text"]
          else:
            text = str(completion)
        else:
          text = completion
        cache[prompt] = {
            "text": text,
            "num_thinking_tokens": 0,
            "cot": "",
        }
      elif "text" in entry:
        cache[prompt] = {
            "text": entry["text"],
            "num_thinking_tokens": entry.get("num_thinking_tokens", 0),
            "cot": entry.get("cot", ""),
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
  """Async request to OpenAI Chat Completions API."""
  client = get_azure_openai_client()
  response = await client.chat.completions.create(
      model=model,
      messages=messages,
      **generation_config
  )
  return {
      "choices": [{"message": {"content": response.choices[0].message.content}}],
      "usage": {
          "prompt_tokens": response.usage.prompt_tokens,
          "completion_tokens": response.usage.completion_tokens,
          "reasoning_tokens": getattr(response.usage, "reasoning_tokens", 0),
      }
  }


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def gemini_request(
    model: str,
    messages: List[Dict[str, str]],
    **generation_config
) -> str:
  """Async request to Google Gemini API using google-genai SDK."""
  # Convert messages to Gemini format
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
  
  # Combine consecutive messages with same role
  combined_messages = []
  for msg in gemini_messages:
    if combined_messages and combined_messages[-1]["role"] == msg["role"]:
      combined_messages[-1]["parts"].extend(msg["parts"])
    else:
      combined_messages.append(msg)
  
  # Create model with system instruction if provided
  model_kwargs = {}
  if system_instruction:
    model_kwargs["system_instruction"] = system_instruction
  
  gen_model = genai.GenerativeModel(model, **model_kwargs)
  gen_config = genai.GenerationConfig(
      temperature=generation_config.get("temperature", 0.0),
      max_output_tokens=generation_config.get("max_tokens", 16384),
  )
  
  # Run in executor since genai doesn't have native async
  loop = asyncio.get_event_loop()
  
  if len(combined_messages) > 1:
    chat = gen_model.start_chat(history=combined_messages[:-1])
    response = await loop.run_in_executor(
        None, 
        lambda: chat.send_message(
            combined_messages[-1]["parts"][0]["text"],
            generation_config=gen_config
        )
    )
  else:
    response = await loop.run_in_executor(
        None,
        lambda: gen_model.generate_content(
            combined_messages[0]["parts"][0]["text"],
            generation_config=gen_config
        )
    )
  return response


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
  
  # Build prompt using tokenizer
  apply_kwargs = {
      "conversation": messages,
      "add_generation_prompt": True,
      "tokenize": False,  # IMPORTANT: Return string, not token IDs
  }
  
  # Add reasoning flag if supported by tokenizer
  if config.enable_reasoning:
    apply_kwargs["enable_reasoning"] = True
    apply_kwargs["add_special_tokens"] = True
  
  try:
    raw_prompt_text = tokenizer.apply_chat_template(**apply_kwargs)
  except TypeError:
    # Fallback if tokenizer doesn't support all kwargs
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
  
  # Parse thinking tokens if model supports reasoning
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
      print(f"\nWARNING: No {config.thinking_end_token} token found in response:\n{response_text}\n")
      cot = response_text
      final_output = ""
  elif "gpt-oss" in model:
    # FIXME
    pass
  else:
    num_thinking_tokens = 0
    cot = ""
    final_output = response_text
  
  return GenerationResult(
      text=final_output.strip(),
      num_thinking_tokens=num_thinking_tokens,
      cot=cot.strip(),
  )


def is_local_model(model_name: str) -> bool:
  """Check if a model should use local inference."""
  return get_local_model_config(model_name) is not None


def is_gpt_model(model_name: str) -> bool:
  """Check if model is a remote GPT model via Azure."""
  return model_name in GPT_COSTS or model_name.startswith("gpt-")
      

async def async_batch_generate(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    max_concurrent: int = 64,
    local_model_config: Optional[LocalModelConfig] = None,
) -> List[GenerationResult]:
  """
  Unified async batch generation for all model types.
  
  Returns list of GenerationResult objects.
  """
  if not batch_messages:
    return []
  
  # Reduce concurrency for Azure OpenAI to avoid WAF blocking
  if is_gpt_model(model_name):
    max_concurrent = min(max_concurrent, 5)
  
  semaphore = asyncio.Semaphore(max_concurrent)
  
  async def generate_one(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, GenerationResult]:
    async with semaphore:
      try:
        # Check local models first (qwen_30b, qwen_4b, gpt_oss_20b, magistral)
        if is_local_model(model_name):
          result = await local_model_request(
              model_name, messages, config=local_model_config, **generation_config
          )
          return idx, result
        
        # Remote GPT models via Azure OpenAI
        elif is_gpt_model(model_name):
          response = await openai_chat_request(model_name, messages, **generation_config)
          text = response["choices"][0]["message"]["content"]
          num_thinking_tokens = response["usage"].get("reasoning_tokens", 0)
          if num_thinking_tokens == 0:
            num_thinking_tokens = response["usage"]["completion_tokens"]
          return idx, GenerationResult(text=text, num_thinking_tokens=num_thinking_tokens)
        
        # Claude models
        elif model_name in CLAUDE_MODELS or model_name.startswith("claude"):
          response = await claude_request(model_name, messages, **generation_config)
          text = response["content"][0]["text"]
          # FIXME
          num_thinking_tokens = None
          return idx, GenerationResult(text=text)
        
        # Gemini models
        elif model_name in GEMINI_MODELS or "gemini" in model_name.lower():
          response = await gemini_request(model_name, messages, **generation_config)
          return idx, GenerationResult(text=response.text, num_thinking_tokens=response.usage_metadata.thoughts_token_count)
        
        else:
          raise ValueError(f"Unknown model: {model_name}")
      
      except Exception as e:
        print(f"Error generating response for index {idx}: {e}")
        raise
  
  tasks = [generate_one(i, msg) for i, msg in enumerate(batch_messages)]
  results_with_idx = await asyncio.gather(*tasks, return_exceptions=True)
  
  processed_results = []
  for result in results_with_idx:
    if isinstance(result, Exception):
      raise result
    processed_results.append(result)
  
  processed_results.sort(key=lambda x: x[0])
  return [result for _, result in processed_results]


def model_call_wrapper(
    model_name: str,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
) -> List[GenerationResult]:
  """
  Wrapper for calling various types of models.
  
  Returns list of GenerationResult objects.
  """
  return asyncio.run(async_batch_generate(
      model_name, batch_messages, generation_config,
      local_model_config=local_model_config,
  ))


def cached_generate(
    batch_prompts: List[List[Dict[str, str]]],
    model_name: str,
    cache: Optional[Dict[str, Any]],
    cache_file: Optional[str],
    generation_config: Dict[str, Any],
    local_model_config: Optional[LocalModelConfig] = None,
) -> Tuple[List[str], List[int], List[str]]:
  """
  Generate a batch of responses from a model, caching responses.

  Args:
    batch_prompts: The batch of prompts.
    model_name: The name of the model to generate from.
    model_url: The URL of the model (for backwards compatibility).
    cache: Cache of LLM responses.
    cache_file: Cache file of LLM responses.
    generation_config: Generation config for LLM.
    local_model_config: Optional config for local models.

  Returns:
    Tuple of (batch_responses, all_num_thinking_tokens, all_cots).
  """
  if cache is None:
    results = model_call_wrapper(
        model_name,
        batch_messages=batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
    )
    batch_responses = [r.text for r in results]
    all_num_thinking_tokens = [r.num_thinking_tokens for r in results]
    all_cots = [r.cot for r in results]
    return batch_responses, all_num_thinking_tokens, all_cots
  
  new_batch_prompts = []
  new_prompt_indices = []
  for i, prompt in enumerate(batch_prompts):
    jsonified_prompt = jsonify_prompt(prompt)
    if jsonified_prompt not in cache:
      new_batch_prompts.append(prompt)
      new_prompt_indices.append(i)
  
  if new_batch_prompts:
    batch_results = model_call_wrapper(
        model_name,
        batch_messages=new_batch_prompts,
        generation_config=generation_config,
        local_model_config=local_model_config,
    )
    
    for prompt, result in zip(new_batch_prompts, batch_results):
      jsonified_prompt = jsonify_prompt(prompt)
      
      cache[jsonified_prompt] = {
          "text": result.text,
          "num_thinking_tokens": result.num_thinking_tokens,
          "cot": result.cot,
      }
      
      if cache_file:
        cache_entry = {
            "prompt": jsonified_prompt,
            "completion": result.text,
            "num_thinking_tokens": result.num_thinking_tokens,
            "cot": result.cot,
        }
        with open(cache_file, "a") as f:
          f.write(json.dumps(cache_entry) + "\n")
  
  batch_responses = []
  all_num_thinking_tokens = []
  all_cots = []
  
  for prompt in batch_prompts:
    jsonified_prompt = jsonify_prompt(prompt)
    cached_value = cache[jsonified_prompt]
    
    if isinstance(cached_value, dict):
      text_output = cached_value.get("text", "")
      num_tokens = cached_value.get("num_thinking_tokens", 0)
      cot = cached_value.get("cot", "")
    elif isinstance(cached_value, tuple):
      text_output = cached_value[0]
      num_tokens = cached_value[1] if len(cached_value) > 1 else 0
      cot = cached_value[2] if len(cached_value) > 2 else ""
    else:
      text_output = str(cached_value)
      num_tokens = 0
      cot = ""
    
    batch_responses.append(text_output)
    all_num_thinking_tokens.append(num_tokens)
    all_cots.append(cot)
  
  return batch_responses, all_num_thinking_tokens, all_cots


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
