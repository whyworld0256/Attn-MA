import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_activations(model, tokenizer, prompts, num_samples, keys_path, values_path, token_length=64, device=torch.device("cuda")):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_states_all_sample = []
    values_all_sample = []
    keys_all_sample = []
    count = 0
    for prompt in tqdm(prompts):
        count += 1
        if count == num_samples+1:
          break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for key in inputs.keys():
            assert inputs[key].shape[1] >= token_length
            inputs[key] = inputs[key][:, :token_length]

        outputs = model.generate(
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
          
          print(layer_cache.keys[0].shape)
          keys_all_layer.append(layer_cache.keys[0].transpose(0,1))    # [num_heads, seq_len, head_dim]
          values_all_layer.append(layer_cache.values[0].transpose(0,1))  # [num_heads, seq_len, head_dim]

        keys_all_layer = torch.cat(keys_all_layer, dim=0)    # [num_heads, total_seq_len, head_dim]
        values_all_layer = torch.cat(values_all_layer, dim=0)  # [num_heads, total_seq_len, head_dim]
        
        keys_all_sample.append(keys_all_layer.unsqueeze(dim=0))     
        values_all_sample.append(values_all_layer.unsqueeze(dim=0)) 

    # attention_scores_all_sample = torch.cat(attention_scores_all_sample, dim=0)  # (num_samples, num_layers, num_heads, num_tokens)
    keys_all_sample = torch.cat(keys_all_sample, dim=0)
    values_all_sample = torch.cat(values_all_sample, dim=0)
    # np.save(score_path, attention_scores_all_sample.cpu().numpy())
    np.save(values_path, values_all_sample.cpu().numpy())
    np.save(keys_path, keys_all_sample.cpu().numpy())


def measure_open_sourced_lms():
    # load model family
    device = torch.device("cuda")
    os.makedirs("results", exist_ok=True)
    ########################################
    llama2_family = ["meta-llama/Llama-2-7b-hf"]
    llama3_family = ["meta-llama/Meta-Llama-3-8B"]
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
        token_length = 1
            
        values_path = f"results/{model_name}/values_token{token_length}.npy"
        keys_path = f"results/{model_name}/keys_token{token_length}.npy"
        with open(file_path, 'r') as f:
          prompts = [json.loads(line)["text"] for line in f]
        num_samples = 50

        measure_activations(model, tokenizer, prompts, num_samples, keys_path, values_path, token_length, device)

        
if __name__ == "__main__":
    measure_open_sourced_lms()