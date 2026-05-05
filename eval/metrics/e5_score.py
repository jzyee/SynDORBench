"""
E5 Score Metric
---------------
Computes semantic similarity between predicted and reference texts
using the E5 embedding model (e.g., intfloat/e5-large-v2).

This metric provides a cosine similarity–based alignment score,
useful for evaluating text-to-text quality (e.g., LVLM caption outputs).

Requirements:
    pip install transformers torch pandas
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import gc
from tqdm import tqdm


# ============================================================
# E5 Scorer Class
# ============================================================
class E5Scorer:
    """
    Wrapper for the E5 embedding model to compute cosine similarity
    between prediction and reference texts.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (default: "intfloat/e5-large-v2").
    device : str, optional
        Computation device ("cuda" or "cpu"). Auto-detects if not provided.
    """

    def __init__(self, model_name: str = "intfloat/e5-large-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------
    # Encode texts into normalized embeddings
    # ------------------------------------------------------------
    def encode(self, texts):
        """
        Encode a list of texts into normalized embeddings.

        Returns
        -------
        torch.Tensor
            Normalized embeddings (shape: [N, D]).
        """
        with torch.no_grad():
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            last_hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).bool()

            # Mean pooling over the valid tokens
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).float()
            emb = F.normalize(pooled, p=2, dim=1)
            return emb.cpu()

    # ------------------------------------------------------------
    # Compute cosine similarity between prediction and reference
    # ------------------------------------------------------------
    def similarity(self, preds, refs):
        """
        Compute cosine similarity between prediction and reference embeddings.

        Parameters
        ----------
        preds : list[str]
        refs : list[str]

        Returns
        -------
        list[float]
            Cosine similarity scores in [-1, 1].
        """
        emb_pred = self.encode(preds)
        emb_ref = self.encode(refs)
        sims = F.cosine_similarity(emb_pred, emb_ref, dim=1).tolist()
        return sims

    # ------------------------------------------------------------
    # Free GPU/CPU memory
    # ------------------------------------------------------------
    def cleanup(self):
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Global E5 Score
# ============================================================
def compute_e5score(pred_path, ref_path,
                    key_pred="prediction", key_ref="action",
                    key_img="img_path",
                    model_name="intfloat/e5-large-v2",
                    device: str = None):
    """
    Compute mean E5-score (cosine similarity) between predictions and references.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)
    merged = pd.merge(pred_df, ref_df, on=key_img, suffixes=("_pred", "_ref"))

    if len(merged) == 0:
        return 0.0

    scorer = E5Scorer(model_name=model_name, device=device)
    preds = merged[key_pred].astype(str).tolist()
    refs = merged[key_ref].astype(str).tolist()

    sims = []
    for i in tqdm(range(0, len(preds), 64), desc="Computing E5 scores"):
        batch_preds = preds[i:i + 64]
        batch_refs = refs[i:i + 64]
        sims.extend(scorer.similarity(batch_preds, batch_refs))

    mean_score = float(sum(sims) / len(sims))
    scorer.cleanup()
    return mean_score


# ============================================================
# E5 Score by Radius
# ============================================================
def compute_e5score_by_radius(pred_path, ref_path,
                              key_pred="prediction", key_ref="action",
                              key_img="img_path", radius_key="radius",
                              radii=[5.4, 10.8, 23.0],
                              model_name="intfloat/e5-large-v2",
                              device: str = None):
    """
    Compute mean E5-score grouped by radius.
    Returns dictionary radius → mean cosine similarity.
    """
    pred_df = pd.read_json(pred_path, lines=True)
    ref_df = pd.read_json(ref_path, lines=True)
    merged = pd.merge(pred_df, ref_df, on=key_img, suffixes=("_pred", "_ref"))

    results = {}
    for r in radii:
        subset = merged[merged[radius_key] == r]
        if len(subset) == 0:
            results[f"{r}r_e5_score"] = None
            continue

        scorer = E5Scorer(model_name=model_name, device=device)
        preds = subset[key_pred].astype(str).tolist()
        refs = subset[key_ref].astype(str).tolist()

        sims = []
        for i in tqdm(range(0, len(preds), 64), desc=f"Radius {r}m E5"):
            batch_preds = preds[i:i + 64]
            batch_refs = refs[i:i + 64]
            sims.extend(scorer.similarity(batch_preds, batch_refs))

        results[f"{r}r_e5_score"] = float(sum(sims) / len(sims)) if sims else None
        scorer.cleanup()

    return results


# ============================================================
# Sanity Check
# ============================================================
if __name__ == "__main__":
    pred_path = "predictions.jsonl"
    ref_path = "annotations.jsonl"

    print("▶ Testing global E5Score...")
    mean_e5 = compute_e5score(pred_path, ref_path)
    print(f"Mean E5Score: {mean_e5:.4f}")

    print("\n▶ Testing E5Score by radius...")
    e5_by_radius = compute_e5score_by_radius(pred_path, ref_path)
    print(e5_by_radius)
