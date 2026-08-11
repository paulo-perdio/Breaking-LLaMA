"""
Evaluation metrics matching the paper's Section 2.3 (Evaluation Method).

BERT F1: semantic similarity between the corrupted-model answer and the
    baseline (pre-corruption) answer, via BERTScore.
ROUGE-1 / ROUGE-2 / ROUGE-L F1: lexical overlap between the same pair.

NOTE: an earlier draft notebook found in the team's files instantiated
RougeScorer(['rouge1','rouge2','rouge3']) — ROUGE-3, not ROUGE-L. That
discrepancy against the published paper's stated methodology is documented
in the repo README ("Known discrepancies"). This module and the values in
results/*.csv both use ROUGE-L (recomputed directly from the recorded
answer text), matching what the published paper's Methods section states.

BERT_MODEL_TYPE is pinned to 'bert-base-uncased' to match the exact backbone
the team's original notebook used (BERTScorer(model_type='bert-base-uncased')).
Without this, bert_score's module-level score() function defaults to
'roberta-large' instead — a different, much larger (~1.4GB vs ~420MB) model
that would silently produce non-comparable numbers.

_bert_scorer is instantiated once at module load and reused for every call —
bert_score's module-level score() convenience function reloads the entire
model from disk on EVERY call instead, which is fine for one-off use but
reloads the model hundreds/thousands of times in batch mode over a large
dataset. Using a persistent BERTScorer instance (same pattern as _rouge
below) avoids that.
"""

from bert_score import BERTScorer
from rouge_score import rouge_scorer

BERT_MODEL_TYPE = "bert-base-uncased"

_bert_scorer = BERTScorer(model_type=BERT_MODEL_TYPE)
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_bert_f1(candidate: str, reference: str) -> float:
    _, _, f1 = _bert_scorer.score([candidate], [reference])
    return float(f1[0])


def compute_rouge_f1(candidate: str, reference: str) -> dict[str, float]:
    scores = _rouge.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def evaluate(candidate: str, reference: str) -> dict[str, float]:
    """Single entry point returning all four metrics used in the paper."""
    result = {"bert_f1": compute_bert_f1(candidate, reference)}
    result.update(compute_rouge_f1(candidate, reference))
    return result