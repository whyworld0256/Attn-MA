import argparse
import types
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# ==========================================
# Custom Attention Overrides (Moved to Global)
# ==========================================
def enable_llama_custom_attention(layer, layer_id, target_p):
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id
    modified_module.target_p = target_p # Store target_p in the module
    
    # Save the original forward to restore later
    modified_module.original_forward = modified_module.forward
    modified_module.forward = types.MethodType(attn_forward, modified_module)
    return modified_module

def restore_llama_attention(layer):
    if hasattr(layer.self_attn, 'original_forward'):
        layer.self_attn.forward = layer.self_attn.original_forward
        del layer.self_attn.original_forward
        del layer.self_attn.target_p

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
        dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

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

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask 

    other_scores = attn_weights[:, :, :, 1:]  # (B, H, Q, K-1)
    S = other_scores.exp().sum(dim=-1, keepdim=True)  # (B, H, Q, 1)

    # Use the target_p stored on the module
    p_target = getattr(module, "target_p", None)
    
    if p_target is not None:
        p_tensor = torch.tensor(p_target, dtype=attn_weights.dtype, device=attn_weights.device)
        eps_val = 1e-6
        p_tensor = torch.clamp(p_tensor, eps_val, 1.0 - eps_val)

        new_s0 = torch.log(p_tensor / (1 - p_tensor)) + torch.log(S)
        attn_weights[:, :, 1:, 0] = new_s0[:, :, 1:, 0]

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# ==========================================
# Core Measurement Function
# ==========================================
def measure_jacobian(
    model, 
    tokenizer, 
    prompts,
    mode,   
    layer_id,
    token_id,
    scaling_factor = 1.0,
    target_p = None,
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
):
    prompt = prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 🚨 Fix: Safety check for sequence length
    seq_len = inputs.input_ids.shape[1]
    print(seq_len)
    if token_id >= seq_len:
        print(f"Warning: token_id {token_id} is out of bounds for prompt of length {seq_len}. Clamping to {seq_len - 1}.")
        token_id = seq_len - 1

    attn_out = []
    hooks = []

    # def rms2_hook(module, input, output):
    #     rms2_in.append(input[0].detach().to("cpu"))
    
    # hooks.append(model.model.layers[layer_id].post_attention_layernorm.register_forward_hook(rms2_hook))


    attn_out = [] # Renamed from rms2_in for clarity

    def attn_output_hook(module, input, output):
        # self_attn returns a tuple: (attn_output, attn_weights, past_key_value)
        # We need the first element, which is the actual attention output tensor.
        attn_out.append(output[0].detach().to("cpu"))
    
    hooks.append(model.model.layers[layer_id].self_attn.register_forward_hook(attn_output_hook))


    if mode == "epsilon" or "scaling":
        layer = model.model.layers[layer_id]
        enable_llama_custom_attention(layer, layer_id, target_p)

    def scaling_input_hook(module, input_tuple):
        hidden_states = input_tuple[0]
        hidden_states[0,token_id,:] /= torch.norm(hidden_states[0,token_id,:],dim=-1)
        hidden_states[0, token_id, :] *= scaling_factor
        return (hidden_states,) + input_tuple[1:]

    hook = model.model.layers[layer_id].input_layernorm.register_forward_pre_hook(scaling_input_hook)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

    hook.remove()
    original_output = attn_out[0].clone() # Clone to avoid overwriting

    res = 0.0
    epochs = 100
    eps = 1e-3
    
    hidden_dim = model.config.hidden_size

    with torch.no_grad(): # Prevent OOM in the loop
        for epoch in range(epochs):
            attn_out.clear()
            
            perturb = torch.randn(hidden_dim)
            perturb_fixed = (perturb / perturb.norm(p=2)).to(device)

            def change_input_hook_perturb(module, input_tuple):
                hidden_states = input_tuple[0]
                hidden_states[0,token_id,:] /= torch.norm(hidden_states[0,token_id,:],dim=-1)
                hidden_states[0, token_id, :] *= scaling_factor
                hidden_states[0, token_id, :] += eps * perturb_fixed
                return (hidden_states,) + input_tuple[1:]

            hook = model.model.layers[layer_id].input_layernorm.register_forward_pre_hook(change_input_hook_perturb)

            outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

            perturb_output = attn_out[0]

            ans = torch.norm(original_output - perturb_output) / eps
            res  = max(res, ans.item()) 

            hook.remove()

    for h in hooks:
        h.remove()
        
    if mode == "epsilon" or "scaling":
        restore_llama_attention(model.model.layers[layer_id])

    return float(f"{res:.4g}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--mode_type', type=str, required=True, choices=["scaling", "epsilon"])
    parser.add_argument('--layer_id', type=int, default=16)
    parser.add_argument('--token_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    llama2_family = ["meta-llama/Llama-2-7b-chat-hf"]# ["meta-llama/Llama-2-7b-hf"] 
    llama31_family = ["meta-llama/Meta-Llama-3.1-8B"]
    llama32_family = ["meta-llama/Llama-3.2-1B"]
    pythia_family = ["EleutherAI/pythia-14m"]
    mistral_family = ["mistralai/Mistral-7B-v0.1"]

    if args.model_type == "llama2_family": model_pool = llama2_family
    elif args.model_type == "llama31_family": model_pool = llama31_family
    elif args.model_type == "llama32_family": model_pool = llama32_family
    elif args.model_type == "mistral_family": model_pool = mistral_family
    elif args.model_type == "pythia_family": model_pool = pythia_family
    else: raise ValueError("Model family is not defined")

    model_path = model_pool[0]
    # prompts = ["Today I am going to study math."]
    prompts = ["This is a simple test sentence"]
    
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        device_map="auto" 
        )

    layer_id = args.layer_id
    token_id = args.token_id
    res_list = []

    if args.mode_type == "scaling":
        # # scaling_factor_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        # scaling_factor_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5]
        # for scaling_factor in tqdm(scaling_factor_list, desc="Scaling"):
        #     res = measure_jacobian(
        #         model, tokenizer, prompts, args.mode_type, layer_id, token_id, 
        #         scaling_factor=scaling_factor
        #     )
        #     res_list.append(res)
        # print("Results:", res_list)

      epsilon =  0.5
      scaling_list = [0.1,1,10,50,100,1000]
      for scaling_factor in tqdm(scaling_list, desc="scaling"):
          p = 1.0 - epsilon
          res = measure_jacobian(
              model, tokenizer, prompts, args.mode_type, layer_id, token_id, scaling_factor = scaling_factor,
              target_p=p
            )
          res_list.append(res)
      print("Results:", res_list)

      

    elif args.mode_type == "epsilon":
        epsilon_list =  [0.1,0.3,0.5,0.7,0.9]
        scaling_factor = 1
        for epsilon in tqdm(epsilon_list, desc="Epsilon"):
            p = 1.0 - epsilon
            res = measure_jacobian(
                model, tokenizer, prompts, args.mode_type, layer_id, token_id, scaling_factor = scaling_factor,
                target_p=p
            )
            res_list.append(res)
        print("Results:", res_list)