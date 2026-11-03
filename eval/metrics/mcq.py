# eval/metrics/mcq.py
"""
MCQ metrics for SynDORBench's 8-way action-recognition evaluation.
--------------------------------------------------------------------

Primary metric is macro-F1, not accuracy: in the original rebuttal fleet,
near-chance accuracy masked a majority-class prediction collapse (one
label predicted ~61% of the time across most models) rather than unbiased
guessing across the 8 options -- macro-F1 surfaces that collapse, plain
accuracy does not. Also reports the dataset-level categorical entropy of
the *predicted* letters (not a per-row softmax entropy) and an 8x8
confusion matrix, both of which track the same collapse.

Consumes the {"img_path", "pred_letter", "probs"} records written by
eval/inference/inference_mcq.py and merges them against the reference
index (which carries "gt_letter", "options", and optionally "radius") on
img_path -- mirroring the merge-on-img_path convention used by every other
metric in this package (accuracy.py, classification.py, ...).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LETTERS = list("ABCDEFGH")


def _prediction_entropy(pred_series, letters=LETTERS):
    """Dataset-level categorical entropy (bits) of the predicted-letter distribution."""
    counts = pred_series.value_counts().reindex(letters, fill_value=0)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _confusion_8x8(y_true, y_pred, letters=LETTERS):
    return confusion_matrix(y_true, y_pred, labels=letters).tolist()


def _report(y_true, y_pred, letters=LETTERS):
    return classification_report(y_true, y_pred, labels=letters, output_dict=True, zero_division=0)


def compute_mcq_metrics(pred_path, ref_path, key_pred="pred_letter", key_ref="gt_letter", rounding=False):
    """
    Compute macro-F1 (primary), accuracy, dataset-level categorical
    prediction-letter entropy, and an 8x8 confusion matrix for the 8-way
    action-MCQ task.

    Unlike eval/metrics/classification.py's normalize_to_label(), pred_letter
    comes straight out of score_options()'s argmax over exactly the 8 valid
    option letters -- there is no ambiguous/-1 case to drop here.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    if pred_df.empty or ref_df.empty:
        merged = pd.DataFrame(columns=[key_ref, key_pred])
    else:
        merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    y_true = merged[key_ref].tolist() if key_ref in merged else []
    y_pred = merged[key_pred].tolist() if key_pred in merged else []

    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "pred_entropy_H": 0.0,
            "confusion_8x8": _confusion_8x8([], []),
        }

    report = _report(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    entropy_H = _prediction_entropy(merged[key_pred])
    confusion = _confusion_8x8(y_true, y_pred)

    if rounding:
        return {
            "macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
            "accuracy": round(acc * 100, 2),
            "pred_entropy_H": round(entropy_H, 4),
            "confusion_8x8": confusion,
        }
    return {
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "accuracy": acc * 100,
        "pred_entropy_H": entropy_H,
        "confusion_8x8": confusion,
    }


def compute_mcq_metrics_by_radius(
    pred_path,
    ref_path,
    radii_list=[5.4, 10.8, 23.0],
    key_pred="pred_letter",
    key_ref="gt_letter",
    key_radius="radius",
    rounding=False,
):
    """
    Per-radius variant of compute_mcq_metrics, flattened like
    classification.py's compute_classification_metrics_by_radius.

    Example output:
    {
        "5.4r_macro_f1": 41.2, "5.4r_accuracy": 44.0, "5.4r_pred_entropy_H": 2.71, "5.4r_confusion_8x8": [...],
        "10.8r_macro_f1": ..., ...
    }
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    def _empty():
        out = {}
        for r in radii_list:
            out.update({
                f"{r}r_macro_f1": None,
                f"{r}r_accuracy": None,
                f"{r}r_pred_entropy_H": None,
                f"{r}r_confusion_8x8": None,
            })
        return out

    if pred_df.empty or ref_df.empty:
        return _empty()

    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))
    if key_radius not in merged.columns:
        # Reference index has no radius column -- nothing to stratify on.
        return _empty()

    results = {}
    for r in radii_list:
        subset = merged[merged[key_radius] == r]
        if subset.empty:
            results.update({
                f"{r}r_macro_f1": None,
                f"{r}r_accuracy": None,
                f"{r}r_pred_entropy_H": None,
                f"{r}r_confusion_8x8": None,
            })
            continue

        y_true = subset[key_ref].tolist()
        y_pred = subset[key_pred].tolist()
        report = _report(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        entropy_H = _prediction_entropy(subset[key_pred])
        confusion = _confusion_8x8(y_true, y_pred)

        if rounding:
            results.update({
                f"{r}r_macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
                f"{r}r_accuracy": round(acc * 100, 2),
                f"{r}r_pred_entropy_H": round(entropy_H, 4),
                f"{r}r_confusion_8x8": confusion,
            })
        else:
            results.update({
                f"{r}r_macro_f1": report["macro avg"]["f1-score"] * 100,
                f"{r}r_accuracy": acc * 100,
                f"{r}r_pred_entropy_H": entropy_H,
                f"{r}r_confusion_8x8": confusion,
            })
    return results
