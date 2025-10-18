import os
import torch
import gc
from PIL import Image
from eval.models.base import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig

class Qwen25VL(BaseModel):
    """Wrapper for Qwen2.5-VL multimodal inference (YAML-configurable, VRAM-safe)."""

    def __init__(self, config):
        super().__init__(config)

        # Default model path — changeable via YAML
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model_path = getattr(config, "model_path", self.model_name)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # --- YAML-driven configuration ---
        device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype_str = str(getattr(config, "dtype", "bfloat16")).lower()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_str, torch.bfloat16)

        device_map = getattr(config, "device_map", "balanced")
        extra_args = getattr(config, "extra_args", {})
        model_kwargs = extra_args.get("model_kwargs", {})

        print(f"🧠 Using device: {device}, dtype: {dtype}, device_map: {device_map}")
        if "max_memory" in model_kwargs:
            print(f"📦 Max memory config: {model_kwargs['max_memory']}")

        # --- Load processor and model ---
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        )

        self.device = device
        print("✅ Qwen2.5-VL model loaded successfully")

    def _free_mem(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, image: Image.Image, prompt: str | None = None) -> str:
        """
        Generate a text response from Qwen2.5-VL given an image and a text prompt.
        """
        if prompt is None:
            prompt = "Describe the content of this image in detail."

        image = image.convert("RGB")

        # Build multimodal message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template to format input text
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,              # ✅ keep as string
            add_generation_prompt=True
        )

        # Prepare inputs for model (text + image)
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        gen_cfg = GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            try:
                outputs = self.model.generate(**inputs, generation_config=gen_cfg)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ OOM detected — retrying on CPU...")
                self._free_mem()
                self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                outputs = self.model.generate(**inputs, generation_config=gen_cfg)

        # Decode only the assistant's final reply
        decoded = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Extract only the part after "assistant"
        # since Qwen2.5-VL includes the chat transcript
        prediction = decoded[0] if decoded else ""
        if "assistant" in prediction:
            prediction = prediction.split("assistant")[-1].strip()

        print("🧾 Cleaned prediction:", prediction)
        self._free_mem()
        return prediction

    def predict(self, img_path: str, prompt: str | None = None) -> str:
        img = Image.open(img_path).convert("RGB")
        return self.generate(img, prompt)