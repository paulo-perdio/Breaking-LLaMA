import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer

"""## Huggie's Face"""

# Reads from the HF_TOKEN environment variable — never hardcode a real token here.
MY_HUGGIEFACE_TOKEN = os.environ["HF_TOKEN"]

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    token = MY_HUGGIEFACE_TOKEN
    )

"""## Load Model"""

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    use_auth_token = MY_HUGGIEFACE_TOKEN
    )

"""## Just Check"""

embedding_layer = model.model.embed_tokens
attention_layers = model.model.layers
final_layer_norm = model.model.norm

print(len(model.model.layers))

layer = model.model.layers[15]

# Attention weights
q_proj_weights = layer.self_attn.q_proj.weight
k_proj_weights = layer.self_attn.k_proj.weight
v_proj_weights = layer.self_attn.v_proj.weight

# MLP weights
gate_proj_weights = layer.mlp.gate_proj.weight
up_proj_weights = layer.mlp.up_proj.weight
down_proj_weights = layer.mlp.down_proj.weight

"""## Function for Breaking"""

def dropout(input: torch.Tensor, p: float = 0.5):
    mask = torch.rand_like(input) > p # creates a bool tensor
    return input * mask

"""## Start Breaking Here"""

drop_prop = 0.05   #5%

for layer in model.model.layers:
    layer.self_attn.q_proj.weight.data.copy_(dropout(layer.self_attn.q_proj.weight, p=drop_prop))
    layer.self_attn.k_proj.weight.data.copy_(dropout(layer.self_attn.k_proj.weight, p=drop_prop))
    layer.self_attn.v_proj.weight.data.copy_(dropout(layer.self_attn.v_proj.weight, p=drop_prop))

    layer.mlp.gate_proj.weight.data.copy_(dropout(layer.mlp.gate_proj.weight, p=drop_prop))
    layer.mlp.up_proj.weight.data.copy_(dropout(layer.mlp.up_proj.weight, p=drop_prop))
    layer.mlp.down_proj.weight.data.copy_(dropout(layer.mlp.down_proj.weight, p=drop_prop))

"""## Test"""

pipe = pipeline(
    'text-generation',
    model=model,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    tokenizer = tokenizer,
    temperature = 0.001,
    repetition_penalty=1.5,
    #topk = 1
)

messages = [
    {"role": "system", "content": "You are useful assistance!"},
    {"role": "user", "content": 'Give me a short answer to the question "What is the capital city of Thailand?"'},
]
outputs = pipe(
    messages,
    max_new_tokens=30
)
print(outputs[0]["generated_text"][-1])