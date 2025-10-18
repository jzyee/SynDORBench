import os
import torch
import gc
from PIL import Image
from eval.models.base import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

class Molmo(BaseModel):
    """Wrapper for Molmo-7B-D multimodal inference (YAML-configurable, VRAM-safe)."""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "allenai/Molmo-7B-D-0924"
        self.model_path = getattr(config, "model_path", self.model_name)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

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

        # --- Load model (trust_remote_code=True enables Molmo custom class) ---
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        )

        self.device = device
        print("✅ Model loaded successfully")

    def _free_mem(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, image: Image.Image, prompt: str | None = None) -> str:
        if prompt is None:
            prompt = "Describe the content of this image in detail."

        image = image.resize((512, 512))
        inputs = self.processor.process(
            images=[image],
            text=prompt,
            return_tensors="pt",
            padding=True,
        )

        model_dtype = next(self.model.parameters()).dtype
        model_device = self.model.device if hasattr(self.model, "device") else self.device

        # Move and cast inputs safely
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(model_device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
                inputs[k] = v

        # Ensure proper shapes
        if "input_ids" in inputs and inputs["input_ids"].ndim == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "attention_mask" in inputs and inputs["attention_mask"].ndim == 1:
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
        if "images" in inputs and inputs["images"].ndim == 3:
            inputs["images"] = inputs["images"].unsqueeze(0)
        elif "pixel_values" in inputs and inputs["pixel_values"].ndim == 3:
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)

        # --- Fix Molmo image tensor shapes ---
        for key in ["images", "pixel_values", "image_input_idx", "image_masks"]:
            if key in inputs and inputs[key].ndim == 2:
                # image_input_idx, image_masks, etc.
                inputs[key] = inputs[key].unsqueeze(0)
            elif key in inputs and inputs[key].ndim == 3 and key in ["images", "pixel_values"]:
                # ensure batch dimension for image tensors
                inputs[key] = inputs[key].unsqueeze(0)    

        print("=== Debug shapes before generate_from_batch ===")
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(f"{k}: {tuple(v.shape)}")

        gen_cfg = GenerationConfig(
            max_new_tokens=256,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            try:
                outputs = self.model.generate_from_batch(
                    batch=inputs,
                    generation_config=gen_cfg,
                    tokenizer=self.processor.tokenizer,
                )
            except torch.cuda.OutOfMemoryError:
                print("⚠️ OOM detected — retrying on CPU...")
                self._free_mem()
                self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                outputs = self.model.generate_from_batch(
                    batch=inputs,
                    generation_config=gen_cfg,
                    tokenizer=self.processor.tokenizer,
                )
        print("=== DEBUG: generate_from_batch output ===")
        print(type(outputs))
        print(outputs)
        
        # Decode tensor -> text
        decoded = self.processor.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        print("🧾 Decoded output:", decoded)
        prediction = decoded[0] if decoded else ""        
        self._free_mem()

        return prediction


    def predict(self, img_path: str, prompt: str | None = None) -> str:
        img = Image.open(img_path).convert("RGB")
        return self.generate(img, prompt)