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

# Assuming these are your local modules
from .load_data import *
from .eval_utils import *

# ==========================================
# Attention Modification Functions
# ==========================================

def enable_llama_custom_attention(layer, layer_id):
    """
    Replace the forward function of LlamaAttention with a custom forward function.
    Saves the original forward method to prevent permanent monkey-patching.
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    
    # Save the original forward method if not already saved
    if not hasattr(modified_module, "original_forward"):
        modified_module.original_forward = modified_module.forward
        
    modified_module.forward = types.MethodType(attn_forward, modified_module)
    return modified_module

def restore_original_attention(model):
    """Restores all modified attention layers back to their original state."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "original_forward"):
            layer.self_attn.forward = layer.self_attn.original_forward
            del layer.self_attn.original_forward

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
    
    value_states = repeat_kv(value, module.num_key_value_groups).clone()
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
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def compute_attention_sink(attention_scores, epsilon):
    num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
    assert num_tokens1 == num_tokens2
    
    # Optional performance boost: Move computation to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_scores = torch.from_numpy(attention_scores).to(device)
    
    ratios = torch.arange(num_tokens1, 0, -1, device=device)[None, None, None, :].expand(
        num_samples, num_layers, num_heads, num_tokens1, num_tokens2
    )
    
    importance_scores = (attention_scores / ratios).sum(dim=-2) 
    metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(0,1,2))
    return metric1 * 100

# ==========================================
# Core Intervention Analysis
# ==========================================

def intervention_analysis(
    model, 
    tokenizer, 
    prompts, 
    num_samples, 
    save_dir, 
    topk,
    epsilon,
    is_eval_ppl,
    scaling_factor = 1, 
    mode = "set_x0",
    token_length=64, 
    add_bos=True, 
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
):

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    rms1_in_all, rms1_out_all = [], []
    attn_in_all, attn_out_all = [], []
    rms2_in_all, rms2_out_all = [], []
    ffn_in_all, ffn_out_all = [], []
    attention_scores_all_sample = []

    rms1_in, rms1_out = [], []
    attn_in, attn_out = [], []
    rms2_in, rms2_out = [], []
    ffn_in, ffn_out = [], []

    hooks = []
    count = 0

    ######### hooks ########## 
    def rms1_hook(module, input, output):
        rms1_in.append(input[0].detach().to("cpu"))
        rms1_out.append(output.detach().to("cpu"))

    def rms2_hook(module, input, output):
        rms2_in.append(input[0].detach().to("cpu"))
        rms2_out.append(output.detach().to("cpu"))

    def ffn_hook(module, input, output):
        ffn_in.append(input[0].detach().to("cpu"))
        ffn_out.append(output.detach().to("cpu"))

    for block in model.model.layers:  
        hooks.append(block.input_layernorm.register_forward_hook(rms1_hook))
        hooks.append(block.post_attention_layernorm.register_forward_hook(rms2_hook))
        hooks.append(block.mlp.register_forward_hook(ffn_hook))

    if scaling_factor is None:
        raise ValueError("Input scaling Error")

    # ==========================================
    # Mode Switcher
    # ==========================================
    if mode == "set_topk_input_zero":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states_copy = input[0][0, 0, :].clone()
            idx = torch.topk(hidden_states_copy.abs(), k=topk).indices
            hidden_states[0, 0, idx] = 0.0
            return (hidden_states,) + input[1:]
        for i in range(2,3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))

    elif mode == "set_topk_output_zero":
        def change_input_hook(module, input, output):
            idx = torch.topk(output[0,0,:].abs(), k=topk).indices
            output[0,0,idx] = 0.0
            return output
        for i in range(2,num_layers):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_hook(change_input_hook))

    elif mode == "turnoff_v0_set_zero":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states_copy = input[0][0, 0, :].clone()
            idx = torch.topk(hidden_states_copy.abs(), k=topk).indices
            hidden_states[0, 0, idx] = 0.0
            return (hidden_states,) + input[1:]
        for i in range(2,3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))
        for layer_id in range(2,num_layers):
            enable_llama_custom_attention(model.model.layers[layer_id], layer_id)

    elif mode == "turnoff_v0_scaling":
        def change_input_hook(module, input, output):
            output[0,0,:] *= scaling_factor
            return output
        hooks.append(model.model.layers[1].register_forward_hook(change_input_hook))
        for layer_id in range(2,num_layers):
            enable_llama_custom_attention(model.model.layers[layer_id], layer_id)

    elif mode == "turnoff_v0_vanilla":
        for layer_id in range(2,32):
            enable_llama_custom_attention(model.model.layers[layer_id], layer_id)

    elif mode == "turnoff_v0_swap":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states_copy = input[0][0, 0, :].clone()
            _, topk_indices = torch.topk(hidden_states_copy.abs(), k=topk)
            all_indices = torch.randperm(4096)
            mask = torch.ones(4096, dtype=bool)
            mask[topk_indices] = False 
            available_indices = all_indices[mask]
            new_indices = available_indices[torch.randperm(len(available_indices))[:topk]]
            
            hidden_states_copy[topk_indices], hidden_states_copy[new_indices] = hidden_states_copy[new_indices], hidden_states_copy[topk_indices]
            hidden_states[0,0,:] = hidden_states_copy
            return (hidden_states,) + input[1:]

        for i in range(2,3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))
        for layer_id in range(2,32):
            enable_llama_custom_attention(model.model.layers[layer_id], layer_id)

    elif mode == "set_swap":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states_copy = input[0][0, 0, :].clone()
            _, topk_indices = torch.topk(hidden_states_copy.abs(), k=topk)
            all_indices = torch.randperm(4096)
            mask = torch.ones(4096, dtype=bool)
            mask[topk_indices] = False 
            available_indices = all_indices[mask]
            new_indices = available_indices[torch.randperm(len(available_indices))[:topk]]
            
            hidden_states_copy[topk_indices], hidden_states_copy[new_indices] = hidden_states_copy[new_indices], hidden_states_copy[topk_indices]
            hidden_states[0,0,:] = hidden_states_copy
            return (hidden_states,) + input[1:]
        for i in range(2,3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))

    elif mode == "set_x0_one_layer":
        def change_input_hook(module, input, output):
            output[0,0,:] *= scaling_factor
            return output
        hooks.append(model.model.layers[1].register_forward_hook(change_input_hook))

    elif mode == "set_x0_all_layers":
        x_0 = torch.from_numpy(np.load(f"./{save_dir}/rms1_in_bos.npy")).to(device)
        x_0_token_0 = x_0[0,2,0] 
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states[:, 0, :] =  x_0_token_0.to(device) * scaling_factor  
            return (hidden_states,) + input[1:]
        for i in range(2,num_layers):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))
  
    elif mode == "turnoff_attn":
        turnoff_layer_id = [0,1]
        def forward_skip_self_attention(self, hidden_states, attention_mask=None,position_ids=None, past_key_values=None, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            attn_output = torch.zeros_like(hidden_states)
            batch_size, seq_len, hidden_size = hidden_states.shape
            attn_weights = torch.zeros(1, self.num_heads, seq_len, seq_len, device=hidden_states.device)
            return attn_output, attn_weights
            
        for i in turnoff_layer_id:
            attn = model.model.layers[i].self_attn
            if not hasattr(attn, "original_forward"):
                attn.original_forward = attn.forward
            attn.forward = forward_skip_self_attention.__get__(attn, type(attn))

    elif mode == "set_zero":
        def change_input_hook(module, input, output):
            output[0,0,:] *= 0.0
            return output
        hooks.append(model.model.layers[1].register_forward_hook(change_input_hook))

    elif mode == "set_random":
        def change_input_hook(module, input):
            hidden_states = input[0]
            hidden_states[:, 0, :] =  torch.randn_like(hidden_states[:, 0, :]).to(device)
            return (hidden_states,) + input[1:]
        for i in range(2,3):
            hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))    
            
    elif mode == "original":
        pass
    else:
        raise ValueError("mode is not defined")

    # ==========================================
    # Execution
    # ==========================================
    try:
        if not is_eval_ppl:
            for prompt in tqdm(prompts):
                count += 1
                if count == num_samples + 1:
                    break
                
                rms1_in.clear(); rms1_out.clear()
                attn_in.clear(); attn_out.clear()
                rms2_in.clear(); rms2_out.clear()
                ffn_in.clear(); ffn_out.clear()
              
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                for key in inputs.keys():
                    assert inputs[key].shape[1] >= token_length
                    inputs[key] = inputs[key][:, :token_length]

                # 🚨 FIX: Added torch.no_grad() to prevent massive memory leak
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=False,
                        use_cache=True,
                        return_dict=True
                    )

                rms1_in_all.append(torch.stack(rms1_in).squeeze(dim=1))
                rms1_out_all.append(torch.stack(rms1_out).squeeze(dim=1))
                rms2_in_all.append(torch.stack(rms2_in).squeeze(dim=1))
                rms2_out_all.append(torch.stack(rms2_out).squeeze(dim=1))
                ffn_in_all.append(torch.stack(ffn_in).squeeze(dim=1))
                ffn_out_all.append(torch.stack(ffn_out).squeeze(dim=1))

                attentions = outputs["attentions"]
                assert len(attentions) == num_layers
                
                attention_scores_all_layer = torch.cat(attentions, dim=0)
                attention_scores_all_sample.append(attention_scores_all_layer.unsqueeze(dim=0))

            ffn_out_all = torch.stack(ffn_out_all)
            rms2_in_all = torch.stack(rms2_in_all)

            if mode == "turnoff_attn":
                rms1_in_all = torch.stack(rms1_in_all)
                norm = (rms2_in_all-rms1_in_all).norm(p=2,dim=-1)[:,0,0].mean(dim=0)
                if norm != 0:
                    raise ValueError("turnoff fail")

            activation_magnitude = rms2_in_all + ffn_out_all

            attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)
            sink_rate = compute_attention_sink(attention_scores_all_sample.detach().to(torch.float32).cpu().numpy(), epsilon=epsilon)

            return (activation_magnitude, sink_rate)

        else:
            ds_list = ["wikitext"]
            res = {}
            seed = 2026
            for ds_name in ds_list:
                ppl = eval_ppl(ds_name, model, tokenizer, seed)
                res[ds_name] = ppl 
                print(f"{ds_name} ppl: {ppl}")
                
            for x,y in res.items():
                print(x, y)
                
            return 1, 1

    finally:
        for h in hooks:
            h.remove()
        
        restore_original_attention(model)