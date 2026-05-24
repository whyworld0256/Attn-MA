import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import types
import warnings
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from torch import nn
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# Assumes these are available in your project structure
from Exps.load_data import *
from Exps.eval_utils import *


def enable_llama_custom_attention(layer, layer_id, custom_k=None, custom_v=None):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    modified_module.custom_k = custom_k
    modified_module.custom_v = custom_v
    modified_module.forward = types.MethodType(attn_forward, modified_module)

    return modified_module


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    custom_k: Optional[torch.Tensor] = None,
    custom_v: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if custom_k is not None:
        key_states[:, :, 0, :] = custom_k
    if custom_v is not None:
        value_states[:, :, 0, :] = custom_v

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:

  input_shape = hidden_states.shape[:-1]
  hidden_shape = (*input_shape, -1, self.head_dim)

  query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
  key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
  value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

  cos, sin = position_embeddings
  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

  if past_key_values is not None:
    # sin and cos are specific to RoPE models; cache_position needed for the static cache
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

  attn_output, attn_weights = eager_attention_forward(
      self,
      query_states,
      key_states,
      value_states,
      attention_mask,
      dropout=0.0 if not self.training else self.attention_dropout,
      scaling=self.scaling,
      custom_k=getattr(self, "custom_k", None),
      custom_v=getattr(self, "custom_v", None),
      **kwargs,
        )

  attn_output = attn_output.reshape(*input_shape, -1).contiguous()
  attn_output = self.o_proj(attn_output)
  return attn_output, attn_weights


def compute_attention_sink(attention_scores, epsilon):
  num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
  assert num_tokens1 == num_tokens2
  
  # NumPy conversion removed as attention_scores is now passed directly as a CPU PyTorch tensor
  ratios = torch.arange(num_tokens1, 0, -1)[None, None, None, :].expand(num_samples, num_layers, num_heads, num_tokens1, num_tokens2).to(attention_scores)
  importance_scores = (attention_scores / ratios).sum(dim=-2) # (num_samples, num_layers, num_heads, num_tokens)
  metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(0,1,2))
  return metric1 * 100
  

def measure_attnsink(
    model, 
    tokenizer, 
    prompts, 
    num_samples, 
    topk,
    epsilon,
    mode="set_x0",
    token_length=64, 
    add_bos=True, 
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
):

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    attention_scores_all_sample = []

    # 1. Load external data ONCE outside the loop to prevent I/O bottlenecks
    original_k, original_v = None, None
    if mode == "original_k":
        original_k = torch.from_numpy(np.load("results/Llama-2-7b-hf/keys_token1.npy")).to(device)
    elif mode == "original_v":
        original_v = torch.from_numpy(np.load("results/Llama-2-7b-hf/values_token1.npy")).to(device)

    # 2. Register the hook ONCE outside the loop
    hooks = []
    if mode in ["h=0", "original_k", "original_v"]:
        def change_input_hook(module, input_args):
            # input_args is a tuple. The first element is hidden_states.
            hidden_states = input_args[0]
            
            # Avoid modifying the original reference unintentionally
            hidden_states_copy = hidden_states[0, 0, :].clone()
            idx = torch.topk(hidden_states_copy.abs(), k=topk).indices
            
            # Modifying in-place is fine for inference
            hidden_states[0, 0, idx] = 0.0
            
            return (hidden_states,) + input_args[1:]

        # Hooking layer 2 (index 2)
        hooks.append(model.model.layers[2].input_layernorm.register_forward_pre_hook(change_input_hook))

    # 3. Main Inference Loop
    for count, prompt in enumerate(tqdm(prompts)):
        if count >= num_samples:
            break

        # Dynamically update the custom_k/custom_v for the current sample
        if mode == "original_k" and original_k is not None:
            # Dynamically handle layer limits based on model config
            for layer_id in range(2, num_layers):
                layer = model.model.layers[layer_id]
                enable_llama_custom_attention(
                    layer, 
                    layer_id, 
                    custom_k=original_k[count, layer_id, :, :]  # count handles sample index
                )
        
        elif mode == "original_v" and original_v is not None:
            for layer_id in range(2, num_layers):
                layer = model.model.layers[layer_id]
                enable_llama_custom_attention(
                    layer, 
                    layer_id, 
                    custom_v=original_v[count, layer_id, :, :] 
                )

        # Tokenization & truncation
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for key in inputs.keys():
            if inputs[key].shape[1] >= token_length:
                inputs[key] = inputs[key][:, :token_length]

        # Model Forward Pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True
            )

        # Aggregate attentions: (num_layers, num_heads, seq_len, seq_len)
        attentions_layer = torch.cat(outputs['attentions'], dim=0)
        attention_scores_all_sample.append(attentions_layer.unsqueeze(0))

    # Clean up hooks after processing all samples to prevent memory leaks
    for h in hooks:
        h.remove()

    # Concatenate all samples: (num_samples, num_layers, num_heads, seq_len, seq_len)
    attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)
    
    # Calculate sink rate (passing the tensor directly, bypassing NumPy overhead)
    attention_scores_tensor = attention_scores_all_sample.detach().to(torch.float32).cpu()
    sink_rate = compute_attention_sink(attention_scores_tensor, epsilon=epsilon)

    print(f"Sink Rate: {sink_rate}")
    return sink_rate


def measure_open_sourced_lms():
    # load model family
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("results", exist_ok=True)
    ########################################
    llama2_family = ["meta-llama/Llama-2-7b-hf"]
    model_pool = llama2_family
    ########################################
    for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        os.makedirs(f"results/{model_name}", exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            # torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        # load data and feed them into LLMs
        file_path = "datasets/probe_valid.jsonl"
            
        with open(file_path, 'r') as f:
          prompts = [json.loads(line)["text"] for line in f]
        
        num_samples = 30
        topk = 5
        epsilon = 0.3
        mode = "original_v"
        token_length = 64

        measure_attnsink(model, tokenizer, prompts, num_samples, topk, epsilon, mode, token_length, device)

        
if __name__ == "__main__":
    measure_open_sourced_lms()