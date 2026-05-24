import os
import time
import json
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
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


def compute_similarity(query, key, layer_id):
    cos1 = 0
    cos2 = 0
    num1 = 0
    num2 = 0
    num_layers, num_heads, seq_len, d = query.shape

    k1 = key[layer_id, :, 0, :]
    for i in range(1, seq_len):
        qt = query[layer_id, :, i, :]
        res = F.cosine_similarity(k1, qt, dim=-1, eps=1e-6).mean(dim=(0))
        cos1 += res
        num1 += 1

    for i in range(1, seq_len):
        for j in range(i, seq_len):
            ki = key[layer_id, :, i, :]
            qj = query[layer_id, :, j, :]
            res = F.cosine_similarity(ki, qj, dim=-1, eps=1e-6).mean(dim=(0))
            cos2 += res
            num2 += 1
  
    res1 = cos1 / num1
    res2 = cos2 / num2
    return res1, res2


def enable_llama_custom_attention(layer, layer_id):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
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
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    value_states[:,:,0,:] = 0.0

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

    self.query_states = query_states
    self.key_states = key_states.repeat_interleave(3, dim=1)
    # self.key_states = key_states.repeat_interleave(4, dim=1)

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
  

def cosine_analysis(
    model, 
    tokenizer, 
    prompts, 
    num_samples, 
    topk,
    epsilon, # Note: epsilon is passed but unused in this logic
    mode="original",
    token_length=64, 
    add_bos=True, 
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
):

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    hooks = []
    count = 0
    attention_scores_all_sample = []
    
    q_list = []
    k_list = []

    def hook_fn(module, input, output):
        q_list.append(module.query_states.detach())
        k_list.append(module.key_states.detach())

    # Process different modes
    if mode == "h=0":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states_copy = input[0][0, 0, :].clone()
            idx = torch.topk(hidden_states_copy.abs(), k=topk).indices
            hidden_states[0, 0, idx] = 0.0
            return (hidden_states,) + input[1:]
        for i in range(2, 3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))

    elif mode == "alpha_h":
        scaling_factor = 0.01
        def change_input_hook(module, input, output):
            output[0, 0, :] *= scaling_factor
            return output
        hooks.append(model.model.layers[1].register_forward_hook(change_input_hook))

    elif mode == "original":
        pass
    else:
        raise ValueError(f"Mode {mode} is not defined")

    # Override attention logic and register hooks dynamically across available layers
    for layer_id in range(2, num_layers):
        layer = model.model.layers[layer_id]
        enable_llama_custom_attention(layer, layer_id)
        
    for layer_id in range(2, num_layers):
        hooks.append(model.model.layers[layer_id].self_attn.register_forward_hook(hook_fn))

    # Analysis Configuration
    target_model_layer = 16
    relative_layer_idx = target_model_layer - 2 

    cos_sink = 0
    cos_non_sink = 0
    
    for prompt in tqdm(prompts):
        count += 1
        if count > num_samples:
            break
            
        q_list = [] # Re-initialize for memory safety
        k_list = []

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for key in inputs.keys():
            assert inputs[key].shape[1] >= token_length
            inputs[key] = inputs[key][:, :token_length]

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True
            )

        query_states = torch.cat(q_list, dim=0)
        key_states = torch.cat(k_list, dim=0)
        
        # Calculate similarity mapping the absolute layer to the relative index of our tensor
        cos1, cos2 = compute_similarity(query_states, key_states, relative_layer_idx)
        cos_sink += cos1 
        cos_non_sink += cos2
    
    # Clean up hooks
    for h in hooks:
        h.remove()

    print(f"Mode: {mode}")
    print(f"Avg Cosine Sink: {cos_sink / num_samples}")
    print(f"Avg Cosine Non-Sink: {cos_non_sink / num_samples}")
    
    return cos_sink / num_samples, cos_non_sink / num_samples