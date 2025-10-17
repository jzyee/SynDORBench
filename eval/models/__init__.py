# eval/models/__init__.py
"""
Model Registry for SynDORBench
------------------------------
This module exposes a unified factory function to load supported model classes.
"""

from eval.models.llavaction import LLavaAction
# from eval.models.internvl3 import InternVL3
# from eval.models.molmo import Molmo


def get_model_class(name: str):
    """
    Return the appropriate model wrapper class based on the name.

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