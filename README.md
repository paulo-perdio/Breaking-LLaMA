# Breaking LLaMA — Structural Robustness of LLaMA-3.2-1B-Instruct

Code and results for our ASSE 2025 paper:

> Chawuthai, R., Thongsawaeng, A., Perdio, J.P.L., Zaw, K.K., Kraichoke, P., Nwe, H.M., & Kertkeidkachorn, N. (2025).
> **Assessing the Effects of Corrupted Parameters in a Large Language Model: A Case Study of LLAMA 3.2 1B.**
> *Proceedings of the 2025 6th Asia Service Sciences and Software Engineering Conference (ASSE 2025)*, Tokyo, Japan.
> https://doi.org/10.1145/3775030.3775040

We stochastically zero out weights in specific transformer components of
Llama-3.2-1B-Instruct (self-attention Q/K/V, feed-forward Gate/Up/Down) and
measure how far the model's output drifts from its uncorrupted baseline,
using BERTScore (semantic) and ROUGE (lexical) F1.

## What this repo's data actually represents (read this before citing numbers)

The paper's headline claims — the model degrading at a 10–15% global
corruption threshold, ROUGE collapsing to zero by 20–30%, feed-forward
components being more failure-prone than self-attention — come from an
**aggregate test over ~1,000 randomly sampled GLUE-QNLI questions**
(Section 2.2 of the paper). That aggregate dataset was never provided to
this repo and isn't reconstructable from what the team has on hand.

What **is** in `results/*.csv` is a **single running example** — the prompt
"What is the capital city of Thailand?" — swept across corruption levels and
layers/matrices. This matches the paper's own Table 1 (which uses the exact
same example for illustration), and matches how the team actually collected
this specific data: one question, tracked by hand across settings, per
person's assigned experiment. It is illustrative, not the statistical
evidence base behind the paper's aggregate figures. Treat single rows and
even overall shape here as anecdote, not proof — see the next section for
a concrete case where this single example doesn't cleanly reproduce a
paper claim.

## The real dataset — added, but not executed at full scale

`data/glue-qnli-1000.csv` is the actual 1,000-question GLUE-QNLI sample the
paper used (Section 2.2) — verified: 1,000 unique rows, no empty fields,
`question`/`sentence`/`index` columns matching QNLI's format.

`run_experiment.py` supports a `--dataset` batch mode implementing the
paper's actual aggregate methodology (Section 2.4: generate all baselines
uncorrupted, corrupt, generate all again, average BERT/ROUGE-L F1 across
every question). A 10-question validation run confirmed this works
end-to-end against the real gated model (see "Small-scale validation run"
below) — the full 1,000-question run was not attempted due to local disk
space and compute time constraints. `results/*.csv` remains the
single-example illustration described below, not the full aggregate.

## Small-scale validation run (n=10)

To confirm the batch pipeline works end-to-end against the real gated
model and real dataset, we ran `p=0.15` on the first 10 questions in
`data/glue-qnli-1000.csv` (`results/test_run_p15.csv`). Averaged over
those 10 questions: BERT F1 ≈ 0.33, ROUGE-1 ≈ 0.11, ROUGE-2 ≈ 0.005,
ROUGE-L ≈ 0.09 — a real degradation pattern, qualitatively consistent
with the direction the paper describes at this corruption level.

This is a pipeline sanity check, not a statistical reproduction of the
paper. Ten questions has no statistical power against the paper's real
~1,000-question aggregate, and the full run was not attempted here due
to local disk space and compute time constraints. Two independent runs
at the same setting (no fixed seed, so generation differs each time)
produced close averages (BERT F1 0.319 vs. 0.332, ROUGE-L 0.088 vs.
0.090) — reassuring for pipeline stability, but still not a substitute
for the real aggregate.

## Headline finding — and a real caveat found while recomputing ROUGE-L

Paper's claim: the Feed-Forward Down-projection matrix is the single most
failure-prone component, with corruption there collapsing ROUGE to zero
before other feed-forward matrices show comparable damage (paper: "~20%").

**What this repo's recomputed ROUGE-L on the single Bangkok example actually
shows** (`results/experiment4_feedforward_last_layer.csv`): Up-projection
reaches 0.000 first (25% corruption), Down-projection reaches 0.000 next
(30%), and Gate-projection never reaches 0.000 at all within the tested
0–50% range (floors out around 0.08–0.13). That's a different ordering than
the paper's narrative, on this one example.

This is very likely just small-sample noise — a single test question has no
statistical power, and the paper's real claim was computed over ~1,000
questions, not one. It is not evidence the paper's aggregate finding is
wrong. But it's a real, checkable discrepancy between what's in this repo
and what the paper's prose claims, and papering over it would be worse than
flagging it. If you extend this repo with a real multi-question aggregate
run, this is the first thing to check.

The broader threshold claim (BERT F1 dropping from the 0.6s toward ~0.25–0.3,
ROUGE collapsing to zero somewhere in the 20–30% range under global
corruption) **does** hold up on the recomputed single-example data in
`experiment1_global_corruption.csv` — ROUGE-L there first hits 0.000 at 30%,
consistent with the paper's stated 20–30% window.

## How this code relates to how the research was actually done

**Important distinction:** the four experiments in the paper were originally
run by hand — each team member had their own Google Colab copy of the same
notebook (`notebooks/original_exploratory_notebook.py`), and ran their
assigned experiment independently by editing which layers/matrices to loop
over and what corruption percentage (`drop_prop`) to use in their own copy,
then recording each row's output into a shared spreadsheet. There was no
shared execution environment, no CLI, and no single automated sweep — the
same code was duplicated across separate Colabs and hand-edited per person
(see the per-experiment "Responsible Person" attributions in the results
files below).

`src/` is a **refactor**, done after the fact, that consolidates the same
underlying logic into one parameterized module (`corrupt.py`) and a CLI
(`run_experiment.py`) that *can* reproduce any of the four experiments
programmatically. It did not exist during the actual research and wasn't
used to generate the published results — it exists so the same logic isn't
duplicated across near-identical files and so someone else can rerun any
experiment without hand-editing constants. If you're asked how the
experiments were actually run, the honest answer is "manually, by each
person in their own Colab copy of the same notebook" — not "via this CLI."

## Repo structure

​```
src/
  corrupt.py           # the corruption function (single implementation)
  metrics.py            # BERTScore + ROUGE evaluation
  run_experiment.py     # CLI reproducing any of the paper's 4 experiments
data/
  glue-qnli-1000.csv    # the real 1,000-question dataset the paper used
results/
  experiment1_global_corruption.csv         # global sweep, 5-100%, John Paul
  experiment2_per_layer_15pct.csv           # per-layer sweep @ 15%, Anon
  experiment2_1_per_layer_10pct.csv         # per-layer sweep @ 10%, Phalat
  experiment3_self_attention_last_layer.csv # Q/K/V @ last layer, 5-50%, Phalat
  experiment4_feedforward_last_layer.csv    # Gate/Up/Down @ last layer, 5-50%, Kaung Khant
  test_run_p15.csv                          # n=10 validation run, see "Small-scale validation run"
notebooks/
  original_exploratory_notebook.py    # the actual notebook used for real runs (token redacted)
  scoring_smoke_test_corrected.py     # a second notebook variant with a scoring bug, fixed here — see "Known discrepancies"
​```

Experiments 2–4 were run through 50% corruption, not the full 0–100% range —
that was a deliberate scope decision by the team, not missing data.

The four experiments in the paper are all the same corruption function
(`src/corrupt.py`) applied with different scope. Originally that meant each
person hand-editing their own Colab copy's loop/percentage; this repo
replaces that duplicated-and-hand-edited process with one parameterized
script:

| Experiment | What it targets | Example command |
|---|---|---|
| 1 — Global sweep | All 6 matrices, all 16 layers | `python -m src.run_experiment --layers all --matrices all --p 0.15` |
| 2 — Per-layer | All 6 matrices, one layer at a time | `python -m src.run_experiment --layers 4 --matrices all --p 0.15` |
| 3 — Self-attention | Q/K/V only | `python -m src.run_experiment --layers 15 --matrices self_attn --p 0.20` |
| 4 — Feed-forward | Gate/Up/Down only | `python -m src.run_experiment --layers 15 --matrices feed_forward --p 0.20` |
| Full aggregate | Any config, all 1,000 questions | add `--dataset data/glue-qnli-1000.csv --output results/my_run.csv` to any of the above |

## Setup

Requires Python 3.9+ (uses builtin generic types like `list[int]`).

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your Hugging Face token to .env
```

Get a token at https://huggingface.co/settings/tokens — never hardcode it
into a script or notebook. `run_experiment.py` and
`scoring_smoke_test_corrected.py` both call `load_dotenv()` on startup, so
a `.env` file in the repo root is picked up automatically — you don't need
to `export` the variable manually. `.env` is already in `.gitignore`, so it
will never be committed.

Corruption is stochastic and unseeded by default, matching how the original
experiments were run — pass `--seed N` to `run_experiment.py` for a
reproducible run instead.

Running any experiment also downloads BERTScore's backbone model
(`bert-base-uncased`, ~420MB) on first use, on top of Llama-3.2-1B itself
(~2.5GB). Make sure you have at least 3–4GB of genuinely free disk space
before starting — a near-full system drive can cause the download to fail
partway through with a confusing error rather than a clear "disk full"
message.

## Known discrepancies (documented for reproducibility, not hidden)

- **Generation hyperparameters:** the published paper states `top_k=1` and
  `repetition_penalty=1.05` were used to stabilize outputs (Section 3.1).
  The original exploratory notebook instead used `temperature=0.001` and
  `repetition_penalty=1.5`. This was never reconciled before submission.
  `run_experiment.py` defaults to the paper's stated values; pass different
  generation kwargs if you need to reproduce the notebook's original runs
  instead.
- **A conflicting early draft was found and not used.** One exploratory
  notebook variant in the team's files instantiated
  `RougeScorer(['rouge1', 'rouge2', 'rouge3'])` — ROUGE-3 (trigram overlap),
  not ROUGE-L — and the team's own raw data collection sheet's column
  headers matched that (labeled "ROUGE-3"). This repo's `metrics.py` and
  `results/*.csv` instead use **ROUGE-L**, matching the published paper's
  stated methodology (Section 2.3) — the `rougeL_f1` values in
  `results/*.csv` were recomputed directly from the recorded `answer` text
  using `RougeScorer(['rougeL'])`, not carried over from that draft.
- **Scoring bug found and fixed, but data itself looks unaffected.** A
  second notebook variant (`notebooks/scoring_smoke_test_corrected.py`
  shows the fix; original had the bug) called
  `scorer.score(["generated_text"], ["generated_text1"])` — passing the
  *string literals* `"generated_text"` / `"generated_text1"` instead of
  the variables holding the real model output and reference answer. As
  written, that call would score two fixed placeholder strings against
  each other on every run, which would produce an identical score
  regardless of corruption level. Our actual `experiment1_global_corruption.csv`
  shows a coherent, varying degradation curve — a pattern a constant-string
  comparison cannot produce — so this bug most likely was not active during
  the runs that generated the published results. Documented and fixed here
  so it isn't silently inherited by anyone building on this notebook.
- **Feed-forward matrix ordering doesn't match the paper on this single
  example** — see "Headline finding" above. Most likely single-sample noise
  against the paper's real ~1,000-question aggregate, not a refutation of it.

## Data quality notes (found while transcribing, not hidden)

- **`experiment3_self_attention_last_layer.csv`**: Q's 5% and 10% rows are
  identical (score and generated text both), and V's 5% row is identical to
  K's 10% row. These read as copy-paste artifacts in the original data
  collection sheet rather than independently reproduced runs — treat those
  specific cells as unverified until re-run.
- **`experiment4_feedforward_last_layer.csv`**: Gate's 35% row shows a sharp,
  unexplained recovery toward baseline quality, sandwiched between 30% and
  40% rows that are both heavily degraded. Possibly a real non-monotonic
  effect, possibly a sampling fluke — not verified either way, flagged for
  a re-run before citing it as a finding.

## License

MIT — see LICENSE.
