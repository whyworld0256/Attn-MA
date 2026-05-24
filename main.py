import os
import time
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from Exps.intervention_analysis import *
from Exps.cosine_analysis import *
from Exps.norm_analysis import *
from Fetch_everything import *
from turnoff_attn import *
from perturb import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--mode_type', type=str, required=False)
    parser.add_argument('--add_bos', action="store_true", help="Turn off/on bos")
    parser.add_argument('--seed',type=int, default=2026, help='Seed for sampling the calibration data.')
    parser.add_argument("--eval_ppl", action="store_true", help="1")
    parser.add_argument("--intervention_analysis", action="store_true", help="1")
    parser.add_argument("--norm_analysis", action="store_true", help="1")
    parser.add_argument("--cosine_analysis", action="store_true", help="1")
    parser.add_argument("--Fetch_everything", action="store_true", help="necessary data")
    parser.add_argument("--turnoff_attn", action="store_true", help="turnoff attn")
    parser.add_argument("--perturb", action="store_true", help="turnoff attn")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    
    #######################################
    gpt_family = ["openai-community/gpt2"]#  ["openai-community/gpt2", "openai-community/gpt2-medium", "openai-community/gpt2-large", "openai-community/gpt2-xl"] 
    llama2_family = ["meta-llama/Llama-2-7b-hf"] # ["meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-chat-hf",  "meta-llama/Llama-2-13b-chat-hf"]
    llama3_family = ["meta-llama/Meta-Llama-3-8B"]# ,"meta-llama/Meta-Llama-3-8B-Instruct"]# "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"] # ["meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
    llama31_family = ["meta-llama/Meta-Llama-3.1-8B"]
    llama32_family = ["meta-llama/Llama-3.2-1B"]
    pythia_family = ["EleutherAI/pythia-14m"]# [f"EleutherAI/pythia-{size}" for size in ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]] 
    opt_family = [f"facebook/opt-{size}" for size in ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b"]]
    qwen_family = ["Qwen/Qwen3-8B"] 
    gemma_family = ["google/gemma-3-1b-it"]  # ["google/gemma-7b"]
    mistral_family = ["mistralai/Mistral-7B-v0.3"]#,"mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1"] # [f"mistralai/Mistral-7B-v0.1", f"mistralai/Mistral-7B-Instruct-v0.1"]
    if args.model_type == "llama2_family":
      model_pool = llama2_family
    elif args.model_type == "llama3_family":
      model_pool = llama3_family
    elif args.model_type == "llama31_family":
      model_pool = llama31_family
    elif args.model_type == "llama32_family":
      model_pool = llama32_family
    elif args.model_type == "mistral_family":
      model_pool = mistral_family
    elif args.model_type == "pythia_family":
      model_pool = pythia_family
    elif args.model_type == "qwen_family":
      model_pool = qwen_family
    elif args.model_type == "gemma_family":
      model_pool = gemma_family
    else:
      raise ValueError("Model is not defined")
    ########################################

    if args.Fetch_everything:
      for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        os.makedirs(f"results/{model_name}", exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
          model_path,
          attn_implementation="eager",
          # torch_dtype=torch.bfloat16,
          device_map="auto"
            )
            
        model.eval()
      
        tokenizer = AutoTokenizer.from_pretrained(
          model_path
          )
        file_path = "datasets/probe_valid.jsonl"
        save_dir = f"results/{model_name}"

        with open(file_path, 'r') as f:
          prompts = [json.loads(line)["text"] for line in f]

        # prompts = ["I study math in the library today",
        #           "She studies math in the library today",
        #           "They analyze data in the company today"]

        # prompts = ["Summer is warm. Winter is cold"]

        token_length = 64
        if args.add_bos:
          tokenizer.add_bos_token = True
        else:
          tokenizer.add_bos_token = False
        measure_activations(model, tokenizer, prompts, save_dir, token_length, tokenizer.add_bos_token, device)

    if args.intervention_analysis:
      for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        os.makedirs(f"Exps/exp_mactivation_attnsink/{model_name}", exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        # torch_dtype=torch.bfloat16,
        device_map="auto"
          )
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
        model_path
        )
        file_path = "datasets/probe_valid.jsonl"
        save_dir = f"results/{model_name}"

        with open(file_path, 'r') as f:
          prompts = [json.loads(line)["text"] for line in f]

        token_length = 64
        num_samples = 30
        scaling_factor = 1
        topk = 5
        epsilon = 0.3
        topk_list = [0,1,2,3,4,5]
        # tokenizer.add_bos_token = False
        activation_magnitude_list = []
        sink_rate_list = []
        scaling_factor_list = [0, 0.0001, 0.001, 0.01, 0.1, 1]
        epsilon_list = [0.3, 0.5,0.8]
        mode = args.mode_type

        for epsilon in epsilon_list:
          activation_magnitude, sink_rate = intervention_analysis(
                model, 
                tokenizer, 
                prompts,
                num_samples,
                save_dir,
                topk,
                epsilon,
                args.eval_ppl,
                scaling_factor,
                mode,
                token_length, 
                tokenizer.add_bos_token, 
                device
                )

          if not args.eval_ppl:
            activation_magnitude_list.append(activation_magnitude)
            sink_rate_list.append(sink_rate)

        if not args.eval_ppl:
          sink_rate_all = torch.stack(sink_rate_list).detach().cpu().numpy()
          save_path = f"Exps/exp_mactivation_attnsink/{model_name}/{mode}.npy"
          np.save(save_path, sink_rate_all)
          
          # Store hidden states
          # activation_magnitude = torch.stack(activation_magnitude_list).detach().cpu().numpy()
          # save_path = f"Exps/exp_mactivation_attnsink/{model_name}/{mode}_activation_magnitude.npy"
          # np.save(save_path,activation_magnitude)

    if args.perturb:
      for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        os.makedirs(f"results/{model_name}", exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        # torch_dtype=torch.bfloat16,
        device_map="auto"
          )
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
        model_path
        )

        file_path = "datasets/probe_valid.jsonl"
        save_dir = f"results/{model_name}"
        with open(file_path, 'r') as f:
          prompts = [json.loads(line)["text"] for line in f]

        mode1 = "original"
        mode2 = "perturb_before_ln"
        mode3 = "perturb_after_ln"
        mode_list = [mode1, mode2, mode3]
        layer_id = 4
        token_id = 8
        for mode in mode_list: 
          perturb_analysis(
              model, 
              tokenizer, 
              save_dir,
              prompts,
              mode, 
              layer_id,
              token_id)

    if args.norm_analysis:
      for model_path in tqdm(model_pool):
          model_name = model_path.split("/")[-1]
          os.makedirs(f"Exps/norm_analysis/{model_name}", exist_ok=True)

          model = AutoModelForCausalLM.from_pretrained(
          model_path,
          attn_implementation="eager",
          torch_dtype=torch.bfloat16,
          device_map="auto"
            )
          
          model.eval()
          tokenizer = AutoTokenizer.from_pretrained(
          model_path
          )
          file_path = "datasets/probe_valid.jsonl"

          with open(file_path, 'r') as f:
            prompts = [json.loads(line)["text"] for line in f]

          token_length = 64
          num_samples = 10
          tokenizer.add_bos_token = True
          mode = args.mode_type

          values_path = f"Exps/norm_analysis/{model_name}/value_{mode}.npy"
          keys_path = f"Exps/norm_analysis/{model_name}/key_{mode}.npy"


          norm_analysis(
              model,
              mode, 
              tokenizer, 
              prompts, 
              num_samples,
              keys_path, 
              values_path, 
              token_length=64, 
              device=torch.device("cuda")
              )

    if args.cosine_analysis:
      for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
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
        mode = args.mode_type
        token_length = 10

        cosine_analysis(model, tokenizer, prompts, num_samples, topk, epsilon, mode, token_length, device)

      
            
    





