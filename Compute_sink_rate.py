import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer



def compute_attention_sink(score_path, epsilon):
  attention_scores = np.load(score_path)
  num_samples, num_layers, num_heads, num_tokens1, num_tokens2 = attention_scores.shape
  assert num_tokens1 == num_tokens2
  attention_scores = torch.from_numpy(attention_scores)
  ratios = torch.arange(num_tokens1, 0, -1)[None, None, None, :].expand(num_samples, num_layers, num_heads, num_tokens1, num_tokens2).to(attention_scores)
  importance_scores = (attention_scores / ratios).sum(dim=-2) # (num_samples, num_layers, num_heads, num_tokens)
  metric1 = (importance_scores > epsilon).to(torch.float).mean(dim=(0,1,2))
  return metric1 * 100



score_path = f"results/Llama-2-7b-hf/attn_no_bos.npy"
print(compute_attention_sink(score_path, epsilon=0.3)[0])
