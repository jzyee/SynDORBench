# eval/models/__init__.py
"""
Model Registry for SynDORBench
------------------------------
This module exposes a unified factory function to load supported model classes.
"""

# from .llavaction import LLavaAction
# from .deepseekvl2tiny import DeepSeekVl2Tiny
from .smolvlm import SmolVLM
# from eval.models.internvl3 import InternVL3
# from eval.models.molmo import Molmo


MODEL_REGISTRY = {
    # "llavaction": LLavaAction,
    # "deepseekvl2tiny": DeepSeekVl2Tiny,
    "smolvlm": SmolVLM,
}

def get_model_class(model_name: str):
    """Get model class by name from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]