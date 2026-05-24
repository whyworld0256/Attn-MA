import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_activations(model, tokenizer, prompts, save_dir, token_length=64, add_bos=True, device = torch.device("cuda") if torch.cuda.is_available() else "cpu"):
  num_layers = model.config.num_hidden_layers
  num_heads = model.config.num_attention_heads

  rms1_in_all = []
  rms1_out_all = []
  attn_in_all = []
  attn_out_all = []
  rms2_in_all = []
  rms2_out_all = []
  ffn_in_all = []
  ffn_out_all = []

  attention_scores_all_sample = []

  rms1_in = []
  rms1_out = []
  attn_in = []
  attn_out = []
  rms2_in = []
  rms2_out = []
  ffn_in = []
  ffn_out = []

  hooks = []
  count = 0

  def rms1_hook(module, input, output):
    rms1_in.append(input[0].detach().to("cpu"))
    rms1_out.append(output.detach().to("cpu"))

  # def attn_hook(module, input, output):
    #   attn_in.append(input[0].detach().to("cpu"))
    #   attn_out.append(output.detach().to("cpu"))

  def rms2_hook(module, input, output):
    rms2_in.append(input[0].detach().to("cpu"))
    rms2_out.append(output.detach().to("cpu"))

  def ffn_hook(module, input, output):
    ffn_in.append(input[0].detach().to("cpu"))
    ffn_out.append(output.detach().to("cpu"))

  for block in model.model.layers:  
    hooks.append(block.input_layernorm.register_forward_hook(rms1_hook))
    # hooks.append(block.self_attn.register_forward_hook(attn_hook))
    hooks.append(block.post_attention_layernorm.register_forward_hook(rms2_hook))
    hooks.append(block.mlp.register_forward_hook(ffn_hook))

  for prompt in tqdm(prompts):
    count += 1
    if count == 2:
      break
    
    rms1_in.clear(); rms1_out.clear()
    attn_in.clear(); attn_out.clear()
    rms2_in.clear(); rms2_out.clear()
    ffn_in.clear(); ffn_out.clear()
  
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for key in inputs.keys():
      if inputs[key].shape[1] >= token_length:
        inputs[key] = inputs[key][:, :token_length]

    outputs = model(
    **inputs,
    output_attentions=True,
    output_hidden_states=False,
    use_cache=True,
    return_dict=True
    )


    rms1_in_all.append(torch.stack(rms1_in).squeeze(dim=1))
    rms1_out_all.append(torch.stack(rms1_out).squeeze(dim=1))
    # attn_in_all.append(torch.stack(attn_in))
    # attn_out_all.append(torch.stack(attn_out))
    rms2_in_all.append(torch.stack(rms2_in).squeeze(dim=1))
    rms2_out_all.append(torch.stack(rms2_out).squeeze(dim=1))
    ffn_in_all.append(torch.stack(ffn_in).squeeze(dim=1))
    ffn_out_all.append(torch.stack(ffn_out).squeeze(dim=1))
  
  for h in hooks:
    h.remove()

  def stack_and_save(name, data):
        data = torch.stack([x.to(torch.float32) for x in data]).numpy()
        if add_bos:
          np.save(f"{save_dir}/{name}_bos.npy", data)
        else:
          np.save(f"{save_dir}/{name}_no_bos.npy", data)

  stack_and_save("rms1_in", rms1_in_all)
  stack_and_save("rms1_out", rms1_out_all)
  # stack_and_save("attn_in", attn_in_all)
  # stack_and_save("attn_out", attn_out_all)
  stack_and_save("rms2_in", rms2_in_all)
  stack_and_save("rms2_out", rms2_out_all)
  stack_and_save("ffn_in", ffn_in_all)
  stack_and_save("ffn_out", ffn_out_all)

  # Store attention scores
  attentions = outputs['attentions']
  if add_bos:
    score_path = f"{save_dir}/attn_bos.npy"
  else:
    score_path = f"{save_dir}/attn_no_bos.npy"

  assert len(attentions) == num_layers
  attention_scores_all_layer = []
  for l in range(num_layers):
    attentions_layer = attentions[l]
    attention_scores_all_layer.append(attentions_layer)
  attention_scores_all_layer = torch.cat(attention_scores_all_layer, dim=0)
  attention_scores_all_sample.append(attention_scores_all_layer.unsqueeze(dim=0))
  attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)  # (num_samples, num_layers, num_heads, num_tokens, num_tokens)
  np.save(score_path, attention_scores_all_sample.detach().to(torch.float32).cpu().numpy())