import pandas as pd
from bert_score import score as bert_score

def compute_bertscore(pred_path, ref_path, key_pred="prediction", key_ref="action", model_type="microsoft/deberta-large-mnli"):
    """
    Compute average BERTScore (F1) between model predictions and reference labels.

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
    model_type : str
        Pretrained model to use for BERTScore.

    Returns
    -------
    float
        Mean BERTScore F1 (0–1).
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)

    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    if len(merged) == 0:
        return 0.0

    preds = merged[key_pred].astype(str).tolist()
    refs = merged[key_ref].astype(str).tolist()

    P, R, F1 = bert_score(preds, refs, lang="en", model_type=model_type, rescale_with_baseline=True, idf=False)
    return float(F1.mean())


def compute_bertscore_by_radius(pred_path, ref_path, key_pred="prediction", key_ref="action",
                                radius_key="radius", radii=[5.4, 10.8, 23.0],
                                model_type="microsoft/deberta-large-mnli"):
    """
    Compute mean BERTScore F1 between predictions and references, grouped by radius.

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
        List of radius values to compute scores for.
    model_type : str
        Pretrained model to use for BERTScore.

    Returns
    -------
    dict
        Mapping of radius → mean BERTScore F1.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)
    merged = pd.merge(pred_df, ref_df, on="img_path", suffixes=("_pred", "_ref"))

    results = {}
    for r in radii:
        subset = merged[merged[radius_key+'_ref'] == r]
        if len(subset) == 0:
            results[r] = None
            continue

        preds = subset[key_pred].astype(str).tolist()
        refs = subset[key_ref].astype(str).tolist()

        P, R, F1 = bert_score(preds, refs, lang="en", model_type=model_type, rescale_with_baseline=True, idf=False)
        results[f'{r}r_bert_score'] = float(F1.mean())

    return results


if __name__ == "__main__":
    """
    Diagnostic test for compute_bertscore() and compute_bertscore_by_radius().
    This verifies that:
      - BERTScore runs correctly with baseline rescaling.
      - Returned values are within expected semantic similarity ranges.
      - The merge and key handling logic are correct.
    """

    import pandas as pd
    import tempfile
    import json

    # -----------------------------------
    # Create sample in-memory JSONL data
    # -----------------------------------
    sample_data_pred = [
        {"img_path": "img_1", "prediction": "a man is breaking into a house", "radius": 5.4},
        {"img_path": "img_2", "prediction": "a person is running across the street", "radius": 10.8},
        {"img_path": "img_3", "prediction": "a car is burning", "radius": 23.0},
    ]

    sample_data_ref = [
        {"img_path": "img_1", "action": "a person is entering the home by force", "radius": 5.4},
        {"img_path": "img_2", "action": "someone is sprinting across the road", "radius": 10.8},
        {"img_path": "img_3", "action": "a vehicle on fire", "radius": 23.0},
    ]

    # Write temporary JSONL files
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f_pred, \
         tempfile.NamedTemporaryFile(mode="w", delete=False) as f_ref:

        for row in sample_data_pred:
            f_pred.write(json.dumps(row) + "\n")
        for row in sample_data_ref:
            f_ref.write(json.dumps(row) + "\n")

        pred_path, ref_path = f_pred.name, f_ref.name

    # -----------------------------------
    # Run your compute_bertscore() function
    # -----------------------------------
    print("\n--- Testing compute_bertscore() ---")
    try:
        mean_score = compute_bertscore(pred_path, ref_path)
        print(f"Mean BERTScore (F1): {mean_score:.4f}")
        if 0.6 < mean_score < 0.95:
            print("✅ BERTScore range looks correct.")
        else:
            print("⚠️  BERTScore too low/high — possible baseline issue or cache corruption.")
    except Exception as e:
        print(f"❌ Error running compute_bertscore(): {e}")

    # -----------------------------------
    # Run your compute_bertscore_by_radius() function
    # -----------------------------------
    print("\n--- Testing compute_bertscore_by_radius() ---")
    try:
        results_by_radius = compute_bertscore_by_radius(pred_path, ref_path)
        for r, v in results_by_radius.items():
            print(f"{r}: {v}")
        if all((v is None or 0.6 < v < 0.95) for v in results_by_radius.values() if v is not None):
            print("✅ All per-radius scores within expected range.")
        else:
            print("⚠️  Some radius scores appear off. Check baseline or text alignment.")
    except Exception as e:
        print(f"❌ Error running compute_bertscore_by_radius(): {e}")