# eval/models/llavanextvideo.py

import os
import gc
import torch
from PIL import Image
from eval.models.base import BaseModel
from transformers import (
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    GenerationConfig,
)

class LLaVANextVideo(BaseModel):
    """LLaVA-NeXT-Video (HF) wrapper with YAML-driven configuration."""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        self.model_path = getattr(config, "model_path", self.model_name)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Load YAML-defined runtime options
        device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype_str = str(getattr(config, "dtype", "bfloat16")).lower()
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(dtype_str, torch.bfloat16)
        device_map = getattr(config, "device_map", "auto")
        extra_args = getattr(config, "extra_args", {}) or {}
        model_kwargs = extra_args.get("model_kwargs", {})

        print(f"🧠 Loading LLaVA-NeXT-Video from {self.model_path}")
        print(f"   device={device} dtype={dtype} device_map={device_map}")

        # ✅ Load the correct processor + model classes
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        )

        self.device = device
        self.model_dtype = next(self.model.parameters()).dtype
        print(f"✅ Model loaded successfully (dtype={self.model_dtype})")

    def _free_mem(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, image: Image.Image | list[Image.Image], prompt: str | None = None) -> str:
        if prompt is None:
            prompt = "Describe the content of this image in detail."

        frames = [img.convert("RGB") for img in (image if isinstance(image, list) else [image])]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video" if len(frames) > 1 else "image"},
                ],
            }
        ]

        # 1. Get text prompt (string)
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # 2. Process combined inputs (text + vision)
        # Use single processor call that handles both modalities
        inputs = self.processor(
            text=text_prompt,
            videos=frames if len(frames) > 1 else None,
            images=frames if len(frames) == 1 else None,
            return_tensors="pt",
            padding=True
        )

        # 3. Move tensors to correct device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        gen_cfg = GenerationConfig(
            max_new_tokens=getattr(self, "max_new_tokens", 512),
            temperature=getattr(self, "temperature", 0.0),
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            try:
                output_ids = self.model.generate(**inputs, generation_config=gen_cfg)
            except torch.cuda.OutOfMemoryError:
                self._free_mem()
                self.model.to("cpu")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to("cpu")
                output_ids = self.model.generate(**inputs, generation_config=gen_cfg)

        decoded = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        text = decoded[0] if decoded else ""
        cleaned = text.split("ASSISTANT:")[-1].strip().splitlines()[0]
        self._free_mem()
        return cleaned

    def predict(self, img_path: str, prompt: str | None = None) -> str:
        img = Image.open(img_path).convert("RGB")
        return self.generate(img, prompt)