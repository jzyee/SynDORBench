# eval/metrics/classification.py
import pandas as pd
from sklearn.metrics import classification_report

def compute_classification_metrics(pred_path, ref_path, key_pred="response", key_ref="label"):
    """
    Compute macro/micro precision, recall, and F1.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)
    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    y_true = merged[key_ref].tolist()
    y_pred = merged[key_pred].tolist()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"]
    }