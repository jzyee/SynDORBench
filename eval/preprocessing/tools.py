from __future__ import annotations
from typing import Tuple, Literal, Dict
from PIL import Image, ImageOps

Policy = Literal["fit", "fill", "stretch"]


def _resample_for_scale(scale: float):
    """
    Use LANCZOS for downscaling (anti-aliased), BICUBIC for upscaling.
    """
    return Image.LANCZOS if scale < 1.0 else Image.BICUBIC

def resize_to_exact(
    img: Image.Image,
    target_hw: Tuple[int, int],
    policy: Policy = "stretch",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Dict]:
    """
    Resize an image to exact (H_tgt, W_tgt) for model inputs.

    Args:
        img: RGB PIL.Image
        target_hw: (H_tgt, W_tgt)
        policy: "fit" | "fill" | "stretch"
        pad_color: letterbox padding color (used only for "fit")

    Returns:
        out_img: PIL.Image of size (W_tgt, H_tgt)
        meta: {
           "policy": str,
           "scale": float,            # uniform scale applied before pad/crop (fit/fill)
           "resized_wh": (w', h'),    # size before pad/crop
           "pad": (l, t, r, b),       # padding pixels (fit); else (0,0,0,0)
           "crop": (l, t, r, b)       # crop box in resized coords (fill); else None
        }
    """
    assert img.mode == "RGB", "Convert to RGB before calling (img.convert('RGB'))."
    H_tgt, W_tgt = map(int, target_hw)
    W, H = img.size  # PIL: (W, H)

    if policy == "stretch":
        out = img.resize((W_tgt, H_tgt), Image.BICUBIC)
        return out, {
            "policy": "stretch",
            "scale": (W_tgt / W, H_tgt / H),  # non-uniform
            "resized_wh": (W_tgt, H_tgt),
            "pad": (0, 0, 0, 0),
            "crop": None,
        }

    if policy == "fit":
        s = min(H_tgt / H, W_tgt / W)
        resample = _resample_for_scale(s)
        Wp, Hp = max(1, int(round(W * s))), max(1, int(round(H * s)))
        img_r = img.resize((Wp, Hp), resample)

        # letterbox pad to exact size
        pad_left = (W_tgt - Wp) // 2
        pad_top  = (H_tgt - Hp) // 2
        pad_right = W_tgt - Wp - pad_left
        pad_bottom = H_tgt - Hp - pad_top
        out = ImageOps.expand(img_r, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)

        return out, {
            "policy": "fit",
            "scale": s,
            "resized_wh": (Wp, Hp),
            "pad": (pad_left, pad_top, pad_right, pad_bottom),
            "crop": None,
        }

    if policy == "fill":
        s = max(H_tgt / H, W_tgt / W)
        resample = _resample_for_scale(s)
        Wp, Hp = max(1, int(round(W * s))), max(1, int(round(H * s)))
        img_r = img.resize((Wp, Hp), resample)

        # center-crop to exact size
        left = max(0, (Wp - W_tgt) // 2)
        top  = max(0, (Hp - H_tgt) // 2)
        right = left + W_tgt
        bottom = top + H_tgt
        out = img_r.crop((left, top, right, bottom))

        return out, {
            "policy": "fill",
            "scale": s,
            "resized_wh": (Wp, Hp),
            "pad": (0, 0, 0, 0),
            "crop": (left, top, right, bottom),
        }

    raise ValueError(f"Unknown policy: {policy}")

# FILE: models/llava_action.py

from models.base import BaseModel, evaluate_model  # Import necessary components

class LlavaActionModel(BaseModel):
    """
    LlavaActionModel extends BaseModel to implement specific evaluation logic.
    """

    def __init__(self, config):
        super().__init__(config)
        # Initialize any LlavaAction-specific parameters here

    def evaluate(self, data_loader):
        """
        Evaluate the LlavaAction model using the base evaluation logic.
        
        Args:
            data_loader: DataLoader providing evaluation data.

        Returns:
            metrics: A dictionary of evaluation metrics.
        """
        # Use the base evaluation function or customize it
        metrics = evaluate_model(self, data_loader)
        return metrics


# Example usage
if __name__ == "__main__":
    from data_loader import get_data_loader  # Import your data loader
    from config import get_config  # Import your configuration

    config = get_config()
    data_loader = get_data_loader(config)

    model = LlavaActionModel(config)
    evaluation_metrics = model.evaluate(data_loader)

    print("Evaluation Metrics:", evaluation_metrics)