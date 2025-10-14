# ============================================================
# eval/metrics/rouge_l.py
# ------------------------------------------------------------
# Computes the ROUGE-L (Longest Common Subsequence) F-measure
# between predictions and ground-truth references.
#
# This version allows flexible column naming for both predictions
# and references — suitable for cross-benchmark reuse.
#
# Usage Example:
#   rouge_l(pred_path, ref_path, pred_col="prediction", ref_col="answer")
#
# Requires: pip install rouge-score
# ============================================================

from rouge_score import rouge_scorer
import json


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def compute_rouge_l(pred_path, ref_path, pred_col="prediction", ref_col="action"):
    """
    Compute the mean ROUGE-L F1 score between predictions and references.

    Parameters
    ----------
    pred_path : str or Path
        Path to model predictions JSONL file.
    ref_path : str or Path
        Path to ground-truth JSONL file.
    pred_col : str, optional
        Column name in the prediction JSONL containing model outputs.
        Default is "prediction".
    ref_col : str, optional
        Column name in the reference JSONL containing target outputs.
        Default is "answer".

    Returns
    -------
    dict
        {"rougeL": mean_score}
    """

    preds = load_jsonl(pred_path)
    refs = load_jsonl(ref_path)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    # Ensure both files have equal length (safe alignment)
    n = min(len(preds), len(refs))

    for i in range(n):
        p, r = preds[i], refs[i]

        pred_text = str(p.get(pred_col, "")).strip().lower()
        ref_text = str(r.get(ref_col, "")).strip().lower()

        if not pred_text or not ref_text:
            continue

        score = scorer.score(ref_text, pred_text)["rougeL"].fmeasure
        scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {"rougeL": mean_score}
