import pandas as pd

def compute_accuracy(pred_path, ref_path, key_pred="prediction", key_ref="human_discernable"):
    """
    Compute accuracy between model predictions and reference labels.

    Parameters
    ----------
    pred_path : str or Path
        Path to JSONL file containing model predictions.
    ref_path : str or Path
        Path to JSONL file containing reference annotations.
    key_pred : str
        Column name of predictions.
    key_ref : str
        Column name of reference labels.

    Returns
    -------
    float
        Accuracy score (0–1)
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    # print('pred dataframe')
    # print(pred_df)
    # print('ref dataframe')
    # print(ref_df)
    # Merge on image path
    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))
    # print(merged)
    correct = (merged[key_pred] == merged[key_ref]).sum()
    total = len(merged)
    return correct / total if total > 0 else 0.0




def compute_accuracy_by_radius(pred_path, ref_path, key_pred="prediction", key_ref="human_discernable", radius_key="radius", radii=[5.4, 10.8, 23.0]):
    """
    Compute accuracy between model predictions and reference labels, grouped by radius.

    Parameters
    ----------
    pred_path : str or Path
        Path to JSONL file containing model predictions.
    ref_path : str or Path
        Path to JSONL file containing reference annotations.
    key_pred : str
        Column name of predictions.
    key_ref : str
        Column name of reference labels.
    radius_key : str
        Column name that indicates radius in the dataset.
    radii : list[float]
        List of radius values to compute accuracy for.

    Returns
    -------
    dict
        Dictionary mapping each radius to its accuracy score (0–1).
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    # Merge on image path (or other unique identifier)
    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    results = {}
    for r in radii:
        subset = merged[merged[radius_key] == r]
        total = len(subset)
        if total > 0:
            correct = (subset[key_pred] == subset[key_ref]).sum()
            results[r] = correct / total
        else:
            results[r] = None  # or 0.0 if you prefer a numeric placeholder

    return results
