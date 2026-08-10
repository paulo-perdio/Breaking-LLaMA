# -*- coding: utf-8 -*-
"""
Corrected reference copy of an exploratory scoring-API smoke test found in
the team's second Colab notebook variant.

Two issues in the original were fixed here, both documented in the main
README's "Known discrepancies" section:

1. The original hardcoded a live Hugging Face token in plaintext. Replaced
   below with an environment-variable read — never hardcode a token.

2. The original's scorer calls passed the STRING LITERALS "generated_text"
   and "generated_text1" (in quotes) instead of the variables of the same
   name holding the actual model output and reference answer. Fixed below
   by removing the quotes.
"""

import os
from typing import cast

import numpy as np
import torch
from bert_score import BERTScorer
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()  # populates os.environ from a local .env file, if one exists

HF_TOKEN = os.environ["HF_TOKEN"]  # set this in your shell / .env — never hardcode

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)


def dropout(input: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    mask = torch.rand_like(input, device=input.device) > p
    return input * mask


drop_prop = 0.05
for layer in model.model.layers:
    layer.self_attn.q_proj.weight.data.copy_(dropout(layer.self_attn.q_proj.weight, p=drop_prop))
    layer.self_attn.k_proj.weight.data.copy_(dropout(layer.self_attn.k_proj.weight, p=drop_prop))
    layer.self_attn.v_proj.weight.data.copy_(dropout(layer.self_attn.v_proj.weight, p=drop_prop))
    layer.mlp.gate_proj.weight.data.copy_(dropout(layer.mlp.gate_proj.weight, p=drop_prop))
    layer.mlp.up_proj.weight.data.copy_(dropout(layer.mlp.up_proj.weight, p=drop_prop))
    layer.mlp.down_proj.weight.data.copy_(dropout(layer.mlp.down_proj.weight, p=drop_prop))

pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    tokenizer=tokenizer,
    temperature=0.001,
    repetition_penalty=1.5,
)

messages = [
    {"role": "system", "content": "You are useful assistance!"},
    {"role": "user", "content": 'Give me a short answer to the question "What is the capital city of Thailand?"'},
]
outputs = pipe(messages, max_new_tokens=30)
generated_text = outputs[0]["generated_text"][-1]["content"]  # actual model output
generated_text1 = "The capital city of Thailand is Bangkok."  # handwritten reference answer
print(generated_text)

bert_scorer = BERTScorer(model_type="bert-base-uncased")
# return_hash defaults to False, so this always returns the plain 3-tuple at
# runtime — cast() tells the type checker that directly instead of asking it
# to structurally prove it against bert_score's imprecise stub.
P, R, F1 = cast(
    "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
    bert_scorer.score([generated_text], [generated_text1]),  # FIXED: variables, not literal strings
)
print(f"BERTScore Precision: {P.mean():.3f}, Recall: {R.mean():.3f}, F1: {F1.mean():.3f}")

rouge_scorer_ = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = rouge_scorer_.score(generated_text1, generated_text)  # FIXED: variables, not literal strings
for key in ("rouge1", "rouge2", "rougeL"):
    print(f"{(key.upper() if key != 'rougeL' else 'ROUGE-L')} Precision: {scores[key].precision:.3f}, Recall: {scores[key].recall:.3f}, F1: {scores[key].fmeasure:.3f}")
