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
from typing import Dict, List, Tuple
import numpy as np
import asyncio

from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
import requests
import tenacity
from tenacity import retry
# import torch
import transformers


ThreadPoolExecutor = futures.ThreadPoolExecutor
pipeline = transformers.pipeline
wait_random_exponential = tenacity.wait_random_exponential
stop_after_attempt = tenacity.stop_after_attempt


if "GOOGLE_API_KEY" in os.environ:
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

client = OpenAI(
    base_url=f"http://0.0.0.0:8011/v1",
    api_key="EMPTY",
)
async_client = AsyncOpenAI(
    base_url=f"http://0.0.0.0:8011/v1",
    api_key="EMPTY",
)
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507-FP8")


def load_cache_file(cache_file):
  # TODO: Need to update this whenever changing the saved stuff
  cache = {}
  if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
      for line in f:
        line = json.loads(line)
        # Check if this is a Qwen-style cache entry with thinking tokens and cot
        if "num_thinking_tokens" in line and "cot" in line:
          cache[line["prompt"]] = (
              line["completion"],
              line["num_thinking_tokens"],
              line["cot"]
          )
        else:
          cache[line["prompt"]] = line["completion"]
  return cache


def jsonify_prompt(prompt):
  return json.dumps(prompt)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(
        multiplier=1, max=60
    ),  # Exponential backoff, random wait time between retries
)
def openai_request(model_url, data):
  """Sends a request to an OpenAI model.

  Args:
    model_url: The model url.
    data: The data to send to the model.

  Returns:
    The response from the model.

  Raises:
    Exception: Any errors in the response
  """
  response = requests.post(model_url, headers=OPENAI_HEADER, json=data)
  try:
    response = response.json()
    assert "choices" in response
  except Exception as e:
    print(response)
    raise e
  return response


def process_gemma_messages(messages):
  """Process messages for Gemma models to ensure proper formatting.
  
  Args:
    messages: List of message dictionaries with 'role' and 'content' keys.
    
  Returns:
    List of processed messages that follow the alternating user/assistant pattern.
  """
  # First, convert system to user and combine consecutive messages
  processed_messages = []
  last_role = None
  
  for i, message in enumerate(messages):
    # Convert system role to user role
    current_role = message["role"]
    if current_role == "system":
      current_role = "user"
      message = {"role": "user", "content": message["content"]}
    
    # Combine consecutive messages with the same role
    if current_role == last_role:
      processed_messages[-1]["content"] += "\n\n" + message["content"]
    else:
      processed_messages.append(message)
      last_role = current_role
  
  # Ensure alternating user/assistant pattern
  final_messages = []
  for i, message in enumerate(processed_messages):
    if i == 0 and message["role"] != "user":
      # If first message is not from user, add a dummy user message
      final_messages.append({"role": "user", "content": "Hello"})
    
    # Ensure no consecutive messages with same role
    if i > 0 and message["role"] == final_messages[-1]["role"]:
      if message["role"] == "user":
        final_messages.append({"role": "assistant", "content": "I understand."})
      else:
        final_messages.append({"role": "user", "content": "Please continue."})
    
    final_messages.append(message)
  
  return final_messages


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(
        multiplier=1, max=60
    ),  # Exponential backoff, random wait time between retries
)
def claude_request(
    model_url,
    data
):
  """Sends a request to a Claude model.

  Args:
    model_url: The model url.
    data: The data to send to the model.

  Returns:
    The response from the model.

  Raises:
    Exception: Any errors in the response
  """
  response = requests.post(model_url, headers=ANTHROPIC_HEADER, json=data)
  try:
    response = response.json()
    assert "content" in response
  except Exception as e:
    print(response)
    raise e
  return response


# @retry(
#     stop=stop_after_attempt(3),
#     wait=tenacity.wait_fixed(5),  # Wait 20 seconds between retries
# )
def model_call_wrapper(
    model_name,
    model_url,
    batch_messages: List[List[Dict[str, str]]],
    generation_config: Dict[str, str],
    parallel_model_calls: bool,
) -> List[Tuple[str, int]]:
  """Wrapper for calling various types of models, including Gemini and OpenAI models."""
  if not batch_messages:
    return []
  def get_batch_responses(get_response):
    if not parallel_model_calls or len(batch_messages) <= 1:
      responses = []
      for messages in batch_messages:
        responses.append(get_response(messages))
      return responses
    else:
      with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
        responses = executor.map(
            get_response,
            batch_messages,
        )
        return list(responses)

  if model_name in GPT_COSTS:
    def get_response(messages):
      data = {
          "model": model_name,
          "messages": messages,
          **generation_config,
      }
      response = openai_request(model_url, data)
      return response
    return get_batch_responses(get_response)
  elif "gemini" in model_name.lower():
    # vertexai
    def get_response(messages):
      model = genai.GenerativeModel(model_url)
      for message in messages:
        if message["role"] == "system":
          message["role"] = "user"
        if "content" in message:
          message["parts"] = message["content"]
          del message["content"]
      chat = model.start_chat(history=messages[:-1])
      return chat.send_message(messages[-1]).text

    return get_batch_responses(get_response)
  elif "gemma" in model_name.lower():
    # Use VLLM server for Gemma models
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
  
  elif "qwen" in model_name.lower():
    # Use async for better throughput with continuous batching
    return asyncio.run(qwen_async_batch_generate(model_name, batch_messages))
  
  elif model_name in CLAUDE_MODELS:
    def get_response(messages):
      for message in messages:
        if message["role"] == "system":
          message["role"] = "user"
      data = {
          "model": model_name,
          "messages": messages,
          **generation_config,
      }
      response = claude_request(model_url, data)
      return response
    return get_batch_responses(get_response)


async def qwen_async_batch_generate(model_name, batch_messages, max_concurrent=128):
  """Generate responses for Qwen models using async for better throughput.
  
  Even though vLLM has --max-num-seqs 16, sending more requests async allows
  vLLM's continuous batching to immediately fill slots as requests complete,
  rather than waiting for all 16 to finish before sending the next batch.
  
  Args:
    model_name: The model name.
    batch_messages: List of message lists to generate responses for.
    max_concurrent: Max concurrent requests (can exceed vLLM's max_num_seqs
                    since vLLM will queue them internally).
  """
  semaphore = asyncio.Semaphore(max_concurrent)
  
  async def generate_one(idx, messages):
    async with semaphore:
      raw_prompt_text = tokenizer.apply_chat_template(
        conversation=messages, add_generation_prompt=True, tokenize=False,
        enable_reasoning=True, add_special_tokens=True
      )
      response = await async_client.completions.create(
        model=model_name,
        prompt=raw_prompt_text,
        logprobs=2,
        echo=False,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
      )
      choice = response.choices[0]
      response_text = choice.text
      if "</think>" in choice.logprobs.tokens:
        num_thinking_tokens = choice.logprobs.tokens.index("</think>")
        cot, final_output = response_text.split("</think>", 1)
      else:
        num_thinking_tokens = len(choice.logprobs.tokens)
        print(f"\nWARNING: Timeout in response\n")
        cot = response_text
        final_output = ""
      return idx, (final_output.strip(), num_thinking_tokens, cot.strip())
  
  tasks = [generate_one(i, msg) for i, msg in enumerate(batch_messages)]
  results_with_idx = await asyncio.gather(*tasks)
  # Sort by original index to maintain order
  results_with_idx.sort(key=lambda x: x[0])
  return [result for _, result in results_with_idx]


def cached_generate(
    batch_prompts,
    model_name,
    model_url,
    cache,
    cache_file,
    generation_config,
    parallel_model_calls,
):
  """Generate a batch of responses from a model, caching responses.

  Args:
    batch_prompts: The batch of prompts.
    model_name: The name of the model to generate from.
    model_url: The URL of the model to generate from.
    cache: cache of LLM responses.
    cache_file: cache file of LLM responses.
    generation_config: generation config for LLM
    parallel_model_calls: whether to make parallel calls to the model

  Returns:
    The batch of responses and the cost of the generation.
  """
  if model_name.startswith("o1"):
    for prompt in batch_prompts:
      for t, turn in enumerate(prompt):
        if turn["role"] == "system":
          prompt[t]["role"] = "user"
    generation_config = {}
  if cache is None:
    return model_call_wrapper(
        batch_prompts,
        model_name,
        model_url,
        generation_config=generation_config,
        parallel_model_calls=parallel_model_calls,
    )
  new_batch_prompts = []
  for prompt in batch_prompts:
    # jsonify prompt
    jsonified_prompt = jsonify_prompt(prompt)
    if jsonified_prompt not in cache:
      new_batch_prompts.append(prompt)
    elif model_name in GPT_COSTS and "choices" not in cache[jsonified_prompt]:
      new_batch_prompts.append(prompt)
  batch_responses = model_call_wrapper(
      model_name,
      model_url,
      batch_messages=new_batch_prompts,
      generation_config=generation_config,
      parallel_model_calls=parallel_model_calls,
  )
  for prompt, (response, num_thinking_tokens, cot) in zip(new_batch_prompts, batch_responses):
    jsonified_prompt = jsonify_prompt(prompt)
    cache[jsonified_prompt] = (response, num_thinking_tokens, cot)
    with open(cache_file, "a") as f:
      f.write(
          json.dumps({
              "prompt": jsonified_prompt,
              "completion": cache[jsonified_prompt][0],
              "num_thinking_tokens": cache[jsonified_prompt][1],
              "cot": cache[jsonified_prompt][2],
          })
          + "\n"
      )
  # throws error to retry after having saved
  assert len(batch_responses) == len(new_batch_prompts)
  batch_responses = []
  cost = 0.0
  all_num_thinking_tokens = []
  all_cots = []
  for prompt in batch_prompts:
    jsonified_prompt = jsonify_prompt(prompt)
    if model_name in GPT_COSTS:
      text_output = cache[jsonified_prompt]["choices"][0]["message"]["content"]
      for token_type in GPT_COSTS[model_name]:
        cost += (
            cache[jsonified_prompt]["usage"][token_type]
            * GPT_COSTS[model_name][token_type]
        )
    elif model_name in CLAUDE_MODELS:
      text_output = cache[jsonified_prompt]["content"][0]["text"]
    else:
      text_output = cache[jsonified_prompt][0]
      all_num_thinking_tokens.append(cache[jsonified_prompt][1])
      all_cots.append(cache[jsonified_prompt][2])
    batch_responses.append(text_output)
  return batch_responses, all_num_thinking_tokens, all_cots
