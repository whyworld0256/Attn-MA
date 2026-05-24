import numpy as np
import torch
model_name_1 = "Llama-2-7b-hf"
model_name_2 = "Meta-Llama-3-8B"
model_name_3 = "Mistral-7B-v0.1"
layer_id = 15
def find_max(model_name):
  x_3 = torch.from_numpy(np.load(f"./results/{model_name}/rms2_in_bos.npy"))
  x_5 = torch.from_numpy(np.load(f"./results/{model_name}/ffn_out_bos.npy"))
  x_6 = (x_3 + x_5)[0,layer_id,0,:]
  values, indices = torch.topk(x_6.abs(),5)
  print(indices)


def find_max_2(model_name):
  x_0 = torch.from_numpy(np.load(f"./results/{model_name}/rms1_out_bos.npy"))
  ans = []
  x = x_0[0,layer_id,0,:]
  values, indices = torch.topk(x.abs(),5)
  print(indices)


def find_all(model_name):
  find_max(model_name)
  find_max_2(model_name)

find_all(model_name_3)