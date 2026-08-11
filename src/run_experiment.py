"""
CLI to reproduce any of the paper's four experiments.

Single-question mode (matches the illustrative examples in results/*.csv
and the paper's Table 1):
  python -m src.run_experiment --layers all --matrices all --p 0.15

Batch mode over the full dataset (matches the paper's actual aggregate
methodology, Section 2.4 — averages BERT/ROUGE-L F1 across many questions):
  python -m src.run_experiment --layers all --matrices all --p 0.15 --dataset data/glue-qnli-1000.csv --output results/my_run.csv

--dataset expects a CSV with a `question` column (index/sentence columns,
if present, are ignored — the paper does not use QNLI's provided answers
or context sentences as input, only the questions themselves; Section 2.2).
Use --limit N to test on a small subset before committing to a full,
slow 1,000-question run.

NOTE on scope: this CLI did not exist during the actual research (see
README "How this code relates to how the research was actually done") and
has not been used to reproduce the paper's real aggregate numbers — running
it requires a personally-approved Hugging Face token for the gated
meta-llama/Llama-3.2-1B-Instruct model and non-trivial GPU time for the
full 1,000-question dataset, neither of which is available in an automated
sandbox. This code makes that reproduction *possible*, not something
already done.

Requires HF_TOKEN in the environment. Reads a .env file automatically if
present (see .env.example) — never hardcode a token in this file.

Corruption is stochastic (torch.rand_like with no fixed seed), matching the
original experiments, so two runs at the same --p will not produce
byte-identical results. Pass --seed for a reproducible run.
"""

import argparse
import csv
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
PROMPT_TEMPLATE = 'Give me a short answer to the question "{question}"'


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


def generate_reply(pipe: Pipeline, question: str) -> str:
    """Run generation and extract the assistant's text reply.

    Isolated in its own function with an explicit return type because
    transformers' pipeline() output is loosely typed (the shape varies by
    task), which otherwise causes type-checker errors on chained indexing
    like `pipe(...)[0]["generated_text"][-1]["content"]` at every call site.
    """
    messages = [
        {"role": "system", "content": "You are useful assistance!"},
        {"role": "user", "content": question},
    ]
    result = pipe(messages, **GENERATION_KWARGS)
    return cast(str, result[0]["generated_text"][-1]["content"])


def load_questions(dataset_path: str, limit: int | None) -> list[str]:
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question" not in (reader.fieldnames or []):
            raise ValueError(f"{dataset_path} has no 'question' column (found: {reader.fieldnames})")
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return [PROMPT_TEMPLATE.format(question=r["question"]) for r in rows]


def run_batch(pipe: Pipeline, model, layers: list[int], matrices: tuple[str, ...], p: float, questions: list[str], output_path: str | None) -> None:
    print(f"Generating {len(questions)} baseline answers (uncorrupted model)...")
    baselines = [generate_reply(pipe, q) for q in questions]

    print(f"Corrupting model: p={p} layers={layers} matrices={matrices}")
    corrupt_model(model, p=p, layers=layers, matrices=matrices)

    print(f"Generating {len(questions)} corrupted answers...")
    all_metrics = []
    rows_out = []
    for question, baseline in zip(questions, baselines):
        corrupted = generate_reply(pipe, question)
        m = evaluate(candidate=corrupted, reference=baseline)
        all_metrics.append(m)
        rows_out.append({"question": question, "baseline": baseline, "corrupted": corrupted, **m})

    # Per the paper's Section 2.4: report the average across all questions.
    keys = all_metrics[0].keys()
    averages = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}
    print(f"Averages over {len(questions)} questions: {averages}")

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Per-question results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="all", help="'all', or comma-separated 0-indexed layer indices")
    parser.add_argument("--matrices", default="all", help="'all', 'self_attn', 'feed_forward', or comma-separated q,k,v,gate,up,down")
    parser.add_argument("--p", type=float, required=True, help="corruption probability, e.g. 0.15")
    parser.add_argument("--question", default=BASELINE_QUESTION, help="single-question mode (default)")
    parser.add_argument("--dataset", default=None, help="path to a CSV with a 'question' column — enables batch mode over many questions")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N rows of --dataset (for a quick test before a full run)")
    parser.add_argument("--output", default=None, help="write per-question batch results to this CSV (batch mode only)")
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

    layers = parse_layers(args.layers)
    matrices = parse_matrices(args.matrices)

    if args.dataset:
        questions = load_questions(args.dataset, args.limit)
        run_batch(pipe, model, layers, matrices, args.p, questions, args.output)
        return

    # Single-question mode (original behavior).
    baseline_output = generate_reply(pipe, args.question)
    corrupt_model(model, p=args.p, layers=layers, matrices=matrices)
    corrupted_output = generate_reply(pipe, args.question)
    metrics = evaluate(candidate=corrupted_output, reference=baseline_output)

    print(f"p={args.p} layers={layers} matrices={matrices}")
    print(f"baseline answer:  {baseline_output}")
    print(f"corrupted answer: {corrupted_output}")
    print(metrics)


if __name__ == "__main__":
    main()