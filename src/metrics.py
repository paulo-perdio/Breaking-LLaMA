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
"""

from bert_score import score as bert_score
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_bert_f1(candidate: str, reference: str, lang: str = "en") -> float:
    _, _, f1 = bert_score([candidate], [reference], lang=lang, verbose=False)
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
