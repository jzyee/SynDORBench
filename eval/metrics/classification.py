# eval/metrics/classification.py
import pandas as pd
from sklearn.metrics import classification_report
import re


def normalize_to_label(val):
        """Map free-form text or mixed outputs to numeric class labels."""
        if pd.isna(val):
            return -1
        # Try numeric first
        try:
            val = int(val)
            if val == 1 or val == 0:
                return val
            else:
                return -1
        except (ValueError, TypeError):
            text = str(val).lower().strip()
            if re.search("1", text) and re.search("0", text):
                return -1
            # Affirmative cues
            elif re.search(r"\b(yes|affirmative|true|correct|positive|1)\b", text):
                return 1
            # Negative cues
            elif re.search(r"\b(no|none|not|absent|empty|0)\b", text):
                return 0
            # Fallback for uncertain / ambiguous responses
            return -1

def compute_classification_metrics(pred_path, ref_path, key_pred="prediction", key_ref="human_discernable", rounding=False):
    """
    Compute macro/micro precision, recall, and F1.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    pred_df[key_pred] = pred_df[key_pred].apply(normalize_to_label)
    

    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    # Drop ambiguous predictions (-1)
    merged = merged[merged[key_pred] != -1]

    y_true = merged[key_ref].tolist()
    y_pred = merged[key_pred].tolist()

    if len(y_true) == 0 or len(y_pred) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    if rounding:
        return {
            "accuracy": round(report["accuracy"] * 100, 2),
            "macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
            "macro_precision": round(report["macro avg"]["precision"] * 100, 2),
            "macro_recall": round(report["macro avg"]["recall"] * 100, 2)
        }
    else:
        return {
            "accuracy": report["accuracy"] * 100,
            "macro_f1": report["macro avg"]["f1-score"] * 100,
            "macro_precision": report["macro avg"]["precision"] * 100,
            "macro_recall": report["macro avg"]["recall"] * 100
        }



def compute_classification_metrics_by_radius(
    pred_path,
    ref_path,
    radii_list=[5.4, 10.8, 23.0],
    key_pred="prediction",
    key_ref="human_discernable",
    key_radius="radius",
    rounding=False,
):
    """
    Compute macro/micro classification metrics (accuracy, precision, recall, F1)
    for each specified radius, returning a single flattened dictionary.

    Example output:
    {
        "24r_accuracy": 82.14,
        "24r_macro_f1": 78.53,
        "24r_macro_precision": 80.27,
        "24r_macro_recall": 77.41,
        "48r_accuracy": 75.91,
        ...
    }

    Parameters
    ----------
    pred_path : str or Path
        Path to JSONL file containing model predictions.
    ref_path : str or Path
        Path to JSONL file containing reference annotations.
    radii_list : list
        List of radius values to evaluate (e.g., [24, 48, 72]).
    key_pred : str
        Column name for predicted labels.
    key_ref : str
        Column name for reference labels.
    key_radius : str
        Column name representing the radius field.

    Returns
    -------
    dict
        Flattened dictionary mapping metrics with per-radius prefixes.
    """
    # Load and merge datasets
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)
    
    pred_df[key_pred] = pred_df[key_pred].apply(normalize_to_label)
    
    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    # Drop ambiguous predictions (-1)
    merged = merged[merged[key_pred] != -1]

    results = {}

    for r in radii_list:
        subset = merged[merged[key_radius] == r]
        if subset.empty:
            results.update({
                f"{r}r_accuracy": None,
                f"{r}r_macro_f1": None,
                f"{r}r_macro_precision": None,
                f"{r}r_macro_recall": None
            })
            continue

        y_true = subset[key_ref].tolist()
        y_pred = subset[key_pred].tolist()

        if len(y_true) == 0 or len(y_pred) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        if rounding:
            results.update({
                f"{r}r_accuracy": round(report["accuracy"] * 100, 2),
                f"{r}r_macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
                f"{r}r_macro_precision": round(report["macro avg"]["precision"] * 100, 2),
                f"{r}r_macro_recall": round(report["macro avg"]["recall"] * 100, 2)
            })
        else:
            results.update({
                f"{r}r_accuracy": report["accuracy"] * 100,
                f"{r}r_macro_f1": report["macro avg"]["f1-score"] * 100,
                f"{r}r_macro_precision": report["macro avg"]["precision"] * 100,
                f"{r}r_macro_recall": report["macro avg"]["recall"] * 100,
            })

    return results
