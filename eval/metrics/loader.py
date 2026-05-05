# eval/metrics/loader.py
"""
Metric Loader for SynDORBench
-----------------------------

This module dynamically maps metric names (as declared in the evaluation config)
to their corresponding computation functions.
"""

from eval.metrics.accuracy import compute_accuracy, compute_accuracy_by_radius
from eval.metrics.classification import compute_classification_metrics, compute_classification_metrics_by_radius
from eval.metrics.rouge import compute_rouge_l
from eval.metrics.bert_score import compute_bertscore, compute_bertscore_by_radius
from eval.metrics.e5_score import compute_e5score, compute_e5score_by_radius
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


    # for classification metrics
    if name in ["accuracy", "acc"]:
        return compute_accuracy
    
    elif name in ['dori_accuracy', 'dori_accuracy']:
        return compute_accuracy_by_radius

    elif name in ["classification", "f1", "macro_f1", "precision", "recall"]:
        return compute_classification_metrics

    elif name in ["classification_by_radius"]:
        return compute_classification_metrics_by_radius

    # for sentiment similarity metrics
    elif name in ["rouge_l"]:
        return compute_rouge_l
    
    elif name in ["bert_score", "bertscore"]:
        return compute_bertscore
    elif name in ["bert_score_by_radius", "bertscore_by_radius"]:
        return compute_bertscore_by_radius
    elif name in ["e5_score", "e5score"]:
        return compute_e5score
    elif name in ["e5_score_by_radius", "e5score_by_radius"]:
        return compute_e5score_by_radius
    
    
    # elif name in ["dori", "dori_score"]:
    #     return compute_dori_score

    else:
        raise ValueError(
            f"❌ Unknown metric '{name}'. "
            f"Available metrics: ['accuracy', 'classification', 'dori']"
        )
