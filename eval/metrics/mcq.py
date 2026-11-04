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

#21998: options are shuffled per question, so a letter (A-H) does not refer
to the same action across rows. Accuracy is invariant to this (gt_letter and
pred_letter are resolved through the SAME row's options list, so the letter
and action comparisons are bijective and numerically identical) -- but
macro-F1/entropy/confusion are NOT, since they average per-class. All three
are therefore computed in ACTION space (resolving letter -> options[idx]
per row) to match the paper's action-recognition macro-F1; accuracy stays
on letters, unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LETTERS = list("ABCDEFGH")
ACTIONS = ["sneak", "dodging", "choking", "kicking", "cover face with hands",
           "stepping over something", "walking", "guard up"]


def _letter_to_action(letter, options):
    """Resolve a predicted/gt letter through this row's own (shuffled) options
    list. Returns "<INVALID>" for an unparseable letter -- sklearn's labels=
    restriction means it contributes a false-negative to the true class
    without polluting any of the 8 real classes."""
    if letter in LETTERS:
        idx = LETTERS.index(letter)
        if idx < len(options):
            return options[idx]
    return "<INVALID>"


def _letters_to_actions(letters, options_list):
    return [_letter_to_action(l, opts) for l, opts in zip(letters, options_list)]


def _prediction_entropy(pred_series, letters=ACTIONS):
    """Dataset-level categorical entropy (bits) of the predicted-class distribution."""
    counts = pred_series.value_counts().reindex(letters, fill_value=0)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _confusion_8x8(y_true, y_pred, letters=ACTIONS):
    return confusion_matrix(y_true, y_pred, labels=letters).tolist()


def _report(y_true, y_pred, letters=ACTIONS):
    return classification_report(y_true, y_pred, labels=letters, output_dict=True, zero_division=0)


def compute_mcq_metrics(pred_path, ref_path, key_pred="pred_letter", key_ref="gt_letter", rounding=False):
    """
    Compute macro-F1 (primary), accuracy, dataset-level categorical
    prediction entropy, and an 8x8 confusion matrix for the 8-way
    action-MCQ task.

    macro-F1/entropy/confusion are computed in ACTION space (#21998):
    pred_letter/gt_letter are resolved through each row's own "options" list
    before scoring, since options are shuffled per question and a bare
    letter does not identify a consistent class across rows. Accuracy stays
    letter-based -- identical value either way (bijective per row).

    Unlike eval/metrics/classification.py's normalize_to_label(), pred_letter
    comes straight out of score_options()'s argmax over exactly the 8 valid
    option letters -- there is no ambiguous/-1 case to drop here.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    if pred_df.empty or ref_df.empty:
        merged = pd.DataFrame(columns=[key_ref, key_pred, "options"])
    else:
        merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    y_true_letter = merged[key_ref].tolist() if key_ref in merged else []
    y_pred_letter = merged[key_pred].tolist() if key_pred in merged else []

    if len(y_true_letter) == 0 or len(y_pred_letter) == 0:
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "pred_entropy_H": 0.0,
            "confusion_8x8": _confusion_8x8([], []),
        }

    options_list = merged["options"].tolist()
    y_true = _letters_to_actions(y_true_letter, options_list)
    y_pred = _letters_to_actions(y_pred_letter, options_list)

    report = _report(y_true, y_pred)
    acc = accuracy_score(y_true_letter, y_pred_letter)
    entropy_H = _prediction_entropy(pd.Series(y_pred))
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
    Same ACTION-space scoring as compute_mcq_metrics (#21998).

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
        for r in (radii_list or []):
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

    # Numeric-coerce the radius column so string-typed radii from another
    # pipeline stage (e.g. "5.4") still match the float radii_list -- mirrors
    # the numeric coercion dev's resolve_radii() applied before the raw `==`.
    # (Radii outside radii_list are intentionally excluded for this fixed-radii
    # benchmark; pass radii_list=None to auto-detect every radius in the data.)
    merged[key_radius] = pd.to_numeric(merged[key_radius], errors="coerce")

    if not radii_list:
        radii_list = sorted(merged[key_radius].dropna().unique().tolist())

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

        y_true_letter = subset[key_ref].tolist()
        y_pred_letter = subset[key_pred].tolist()
        options_list = subset["options"].tolist()
        y_true = _letters_to_actions(y_true_letter, options_list)
        y_pred = _letters_to_actions(y_pred_letter, options_list)

        report = _report(y_true, y_pred)
        acc = accuracy_score(y_true_letter, y_pred_letter)
        entropy_H = _prediction_entropy(pd.Series(y_pred))
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
