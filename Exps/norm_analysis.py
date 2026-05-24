import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_kv_norm(hidden_states, device):
    num_samples, num_layers, num_heads, num_tokens, dim = hidden_states.shape
    # hidden_states = torch.from_numpy(hidden_states).to(device)
    split_size = 5
    hidden_states_split = torch.split(hidden_states, split_size)
    all_norms = []
    for hidden_states in hidden_states_split:
        norm = hidden_states.norm(p=2, dim=-1)
        all_norms.append(norm)
    all_norms = torch.cat(all_norms, dim=0)
    return all_norms.mean(dim=(0, 2))  # (num_samples, num_layers, num_heads, num_tokens) -> (num_layers, num_tokens)


def norm_analysis(
  model, 
  mode, 
  tokenizer, 
  prompts, 
  num_samples,
  keys_path, 
  values_path, 
  token_length=50, 
  device=torch.device("cuda")):

  num_layers = model.config.num_hidden_layers
  num_heads = model.config.num_attention_heads
  values_all_sample = []
  keys_all_sample = []
  hooks = []
  count = 0

  if mode == "set_topk_input_zero":
    topk = 5
    def change_input_hook(module,input):
      hidden_states = input[0]
      hidden_states_copy = input[0][0, 0, :].clone()
      idx = torch.topk(hidden_states_copy.abs(), k=topk).indices
      hidden_states[0, 0, idx] = 0.0

      return (hidden_states,) + input[1:]
    for i in range(2,3):
      hooks.append(model.model.layers[i].input_layernorm.register_forward_pre_hook(change_input_hook))

  elif mode == "set_topk_output_zero":
    topk = 3072
    def change_input_hook(module,input,output):
      idx = torch.topk(output[0,0,:].abs(), k=topk).indices
      output[0,0,idx] = 0.0
      return output
    for i in range(2,num_layers):
      hooks.append(model.model.layers[i].input_layernorm.register_forward_hook(change_input_hook))
  elif mode == "original":
    pass
  else:
    raise ValueError("Not a mode")

  for prompt in tqdm(prompts):
    count+=1
    if count == num_samples+1:
      break
    else:
      inputs = tokenizer(prompt, return_tensors="pt").to(device)
      for key in inputs.keys():
        assert inputs[key].shape[1] >= token_length
        inputs[key] = inputs[key][:, :token_length]

      outputs = model(
          **inputs,
          output_attentions=True,
          output_hidden_states=True,
          use_cache=True,
          return_dict_in_generate=True,
          max_new_tokens=1
      )
        
      cache = outputs["past_key_values"]
      values_all_layer = []
      keys_all_layer = []
      
      for layer_cache in cache.layers:
          
        keys_all_layer.append(layer_cache.keys[0])    # [num_heads, seq_len, head_dim]
        values_all_layer.append(layer_cache.values[0])  # [num_heads, seq_len, head_dim]

      keys_all_layer = torch.stack(keys_all_layer, dim=0)    # [num_heads, total_seq_len, head_dim]
      values_all_layer = torch.stack(values_all_layer, dim=0)  # [num_heads, total_seq_len, head_dim]
      
      # print(keys_all_layer.shape)
      keys_all_sample.append(keys_all_layer.unsqueeze(dim=0))     
      values_all_sample.append(values_all_layer.unsqueeze(dim=0)) 

  for h in hooks:
    h.remove()

  keys_all_sample = torch.cat(keys_all_sample, dim=0)
  values_all_sample = torch.cat(values_all_sample, dim=0)
  np.save(values_path, values_all_sample.detach().to(torch.float32).cpu().numpy())
  np.save(keys_path, keys_all_sample.detach().to(torch.float32).cpu().numpy())

