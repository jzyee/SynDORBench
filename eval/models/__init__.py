# eval/models/__init__.py
"""
Model Registry for SynDORBench
------------------------------
This module exposes a unified factory function to load supported model classes.
"""

from eval.models.qwen import Qwen25VL
from eval.models.molmo import Molmo
from eval.models.llavanextvideo import LLaVANextVideo
# from eval.models.internvl3 import InternVL3
# from eval.models.molmo import Molmo


MODEL_REGISTRY = {
    "qwen": Qwen25VL,
    "molmo": Molmo,
    "llava_next_video": LLaVANextVideo  
}

<<<<<<< Updated upstream
    Example:
        model_class = get_model_class("llavaction")
        model = model_class(model_cfg)
    """
    name = name.lower()
    if "llavaction" in name:
        return LLavaAction
    if "llava_mini" in name or "llava-mini" in name:
        from eval.models.llava_mini import LLavaMini
        return LLavaMini
    # elif "internvl3" in name:
    #     return InternVL3
    # elif "molmo" in name:
    #     return Molmo
    else:
        raise ValueError(f"❌ Unknown model name: {name}. "
                         f"Available models: ['llavaction', 'internvl3', 'molmo']")
=======
def get_model_class(model_name: str):
    """Get model class by name from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
>>>>>>> Stashed changes
