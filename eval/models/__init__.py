# eval/models/__init__.py
"""
Model Registry for SynDORBench
------------------------------
This module exposes a unified factory function to load supported model classes.
"""


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
        from eval.models.llavaction import LLavaAction
        return LLavaAction
    # elif "llava_mini" in name or "llava-mini" in name:
    #     from eval.models.llava_mini import LLavaMini
    #     return LLavaMini
    elif "llava_next_video" in name:
        from eval.models.llava_next_video import LlavaNextVideo
        return LlavaNextVideo
    elif "internvl3" in name:
        from eval.models.internvl3 import InternVL3
        return InternVL3
    elif "qwen3vl" in name:
        from eval.models.qwen3vl import Qwen3VL
        return Qwen3VL
    elif "qwen2.5-vl" in name or "qwen2_5_vl" in name:
        from eval.models.qwen2_5_vl import Qwen2_5_VL
        return Qwen2_5_VL
    elif "gemma3" in name or "gemma-3" in name:
        if "gemma3n" in name or "gemma-3n" in name:
            from eval.models.gemma3n import Gemma3n
            return Gemma3n
        else:
            from eval.models.gemma3 import Gemma3
            return Gemma3
    elif "smolvlm" in name:
        from eval.models.smolVLM import SmolVLM
        return SmolVLM
    
    elif "phi4" in name or "phi-4" in name:
        from eval.models.phi4 import Phi4
        return Phi4
    elif "molmo" in name:
        from eval.models.molmo import Molmo
        return Molmo
    elif "perceptionlm" in name or "perception-lm" in name:
        from eval.models.perceptionLM import PerceptionLM
        return PerceptionLM
    elif "yolov11" in name:
        from eval.models.yolo import YOLOHumanDetector
        return YOLOHumanDetector
    elif "o4_mini" in name or "o4-mini":
        from eval.models.gpt_4o_mini import OpenAI4oMini
        return OpenAI4oMini
    
    else:
        raise ValueError(f"❌ Unknown model name: {name}. "
                         f"Available models: ['llavaction', 'internvl3', 'molmo']")