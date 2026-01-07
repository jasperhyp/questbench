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

"""Base class for evaluators."""

from model_utils import CLAUDE_MODELS
from model_utils import GPT_COSTS
from model_utils import load_cache_file
from transformers import pipeline

class Evaluator:
  """Base class for evaluators.

  Attributes:
    model_name: name of LLM to evaluate
    generation_config: generation config for LLM
    model_url: model url of LLM
    cache: cache of LLM responses
    cache_file: cache file of LLM responses
    use_cot: whether to use CoT or not
    fs_samples: number of few-shot samples to use
    eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
    model_role_name: role name for the model
    parallel_model_calls: whether to make parallel calls to the model
    vllm_port: port for the VLLM server
  """

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
      model_role_name: str = "assistant",
      # parallel_model_calls: bool = True,
      vllm_port: int = 8011,
      **kwargs,
  ):
    self.model_name = model_name
    self.generation_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
    }
    if "gemini" in self.model_name:
      self.model_url = self.model_name
    elif self.model_name in GPT_COSTS:
      self.generation_config = {
          # "temperature": 0.0,
          "max_completion_tokens": 16384,
          # "top_p": 1.0,
      }
    elif self.model_name in CLAUDE_MODELS:
      self.generation_config = {
          "temperature": 0.0,
          "max_tokens": 16384,
      }
    elif "qwen" in self.model_name:
      if self.model_name == "qwen_30b":
        self.model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
      elif self.model_name == "qwen_4b":
        self.model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"
    elif self.model_name == "magistral":
      self.model_name = "MistralAI/Magistral-Small-2507"
    elif self.model_name == "gpt_oss_20b":
      self.model_name = "openai/gpt-oss-20b"
    else:
      pass
    self.cache = cache
    self.cache_file = cache_file
    if cache is None and cache_file is not None:
      self.cache = load_cache_file(cache_file)
      print(f"Loaded {len(self.cache)} entries from {cache_file}")
    self.use_cot = use_cot
    self.fs_samples = fs_samples
    self.eval_mode = eval_mode
    self.model_role_name = model_role_name
    # self.parallel_model_calls = parallel_model_calls
    self.vllm_port = vllm_port
    
    self.use_invalid_facts_sets = kwargs.get("use_invalid_facts_sets", False)
