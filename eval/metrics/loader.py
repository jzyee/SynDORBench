# eval/metrics/loader.py
"""
Metric Loader for SynDORBench
-----------------------------

This module dynamically maps metric names (as declared in the evaluation config)
to their corresponding computation functions.
"""

from eval.metrics.accuracy import compute_accuracy
from eval.metrics.classification import compute_classification_metrics
from eval.metrics.rouge import compute_rouge_l
# from eval.metrics.dori_metrics import compute_dori_score


def get_metric_function(name: str):
    """
    Retrieve a metric function based on its name.

    Parameters
    ----------
    name : str
        Name of the metric (e.g., 'accuracy', 'classification', 'dori').

    Returns
    -------
    callable
        The corresponding metric computation function.
    """
    name = name.lower().strip()

    if name in ["accuracy", "acc"]:
        return compute_accuracy

    elif name in ["classification", "f1", "macro_f1", "precision", "recall"]:
        return compute_classification_metrics


    elif name in ["rouge_l"]:
        return compute_rouge_l
    
    # elif name in ["dori", "dori_score"]:
    #     return compute_dori_score

    else:
        raise ValueError(
            f"❌ Unknown metric '{name}'. "
            f"Available metrics: ['accuracy', 'classification', 'dori']"
        )
