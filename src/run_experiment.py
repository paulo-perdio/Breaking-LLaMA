"""
CLI to reproduce any of the paper's four experiments.

Usage examples
---------------
# Experiment 1: global sweep, all layers/matrices, single corruption level
python -m src.run_experiment --layers all --matrices all --p 0.15

# Experiment 2: single layer (0-indexed; paper's "Layer 5" == --layers 4)
python -m src.run_experiment --layers 4 --matrices all --p 0.15

# Experiment 3: self-attention only, last layer
python -m src.run_experiment --layers 15 --matrices q,k,v --p 0.20

# Experiment 4: feed-forward only, last layer
python -m src.run_experiment --layers 15 --matrices gate,up,down --p 0.20

Requires HF_TOKEN in the environment. Reads a .env file automatically if
present (see .env.example) — never hardcode a token in this file.

Corruption is stochastic (torch.rand_like with no fixed seed), matching the
original experiments, so two runs at the same --p will not produce
byte-identical results. Pass --seed for a reproducible run.
"""

import argparse
import os
from typing import cast

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from src.corrupt import ALL_MATRICES, FEED_FORWARD, SELF_ATTENTION, corrupt_model
from src.metrics import evaluate

load_dotenv()  # populates os.environ from a local .env file, if one exists

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LAYERS = 16

# NOTE: the paper's Section 3.1 states generation used top_k=1 and
# repetition_penalty=1.05 to stabilize outputs. The original exploratory
# notebook used temperature=0.001 / repetition_penalty=1.5 instead — that
# mismatch was never reconciled before publication. Defaults below follow
# the PAPER, not the original notebook. Flag if you want the notebook's
# settings reproduced instead.
GENERATION_KWARGS = dict(top_k=1, repetition_penalty=1.05, max_new_tokens=30)

BASELINE_QUESTION = 'Give me a short answer to the question "What is the capital city of Thailand?"'


def parse_matrices(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return ALL_MATRICES
    if raw == "self_attn":
        return SELF_ATTENTION
    if raw == "feed_forward":
        return FEED_FORWARD
    return tuple(raw.split(","))


def parse_layers(raw: str) -> list[int]:
    if raw == "all":
        return list(range(NUM_LAYERS))
    return [int(x) for x in raw.split(",")]


def generate_reply(pipe: Pipeline, messages: list[dict[str, str]]) -> str:
    """Run generation and extract the assistant's text reply.

    Isolated in its own function with an explicit return type because
    transformers' pipeline() output is loosely typed (the shape varies by
    task), which otherwise causes type-checker errors on chained indexing
    like `pipe(...)[0]["generated_text"][-1]["content"]` at every call site.
    """
    result = pipe(messages, **GENERATION_KWARGS)
    return cast(str, result[0]["generated_text"][-1]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="all", help="'all', or comma-separated 0-indexed layer indices")
    parser.add_argument("--matrices", default="all", help="'all', 'self_attn', 'feed_forward', or comma-separated q,k,v,gate,up,down")
    parser.add_argument("--p", type=float, required=True, help="corruption probability, e.g. 0.15")
    parser.add_argument("--question", default=BASELINE_QUESTION)
    parser.add_argument("--seed", type=int, default=None, help="optional, for reproducible corruption across runs")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file (see .env.example) — never hardcode it.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=hf_token)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
    messages = [
        {"role": "system", "content": "You are useful assistance!"},
        {"role": "user", "content": args.question},
    ]
    # Per the paper's methodology (Section 2.2): the baseline is the model's
    # own uncorrupted response, not a handwritten reference answer.
    baseline_output = generate_reply(pipe, messages)

    layers = parse_layers(args.layers)
    matrices = parse_matrices(args.matrices)
    corrupt_model(model, p=args.p, layers=layers, matrices=matrices)

    corrupted_output = generate_reply(pipe, messages)

    metrics = evaluate(candidate=corrupted_output, reference=baseline_output)

    print(f"p={args.p} layers={layers} matrices={matrices}")
    print(f"baseline answer:  {baseline_output}")
    print(f"corrupted answer: {corrupted_output}")
    print(metrics)


if __name__ == "__main__":
    main()
