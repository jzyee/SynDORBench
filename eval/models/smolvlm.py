"""
SmolVLM-Instruct model wrapper for SynDORBench evaluation framework.

Requirements:
- transformers>=4.45.0
- torch>=2.0
- pillow
- accelerate
"""

from .base import BaseModel
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class SmolVLM(BaseModel):
    """Wrapper for HuggingFaceTB/SmolVLM-Instruct (2B) vision-language model."""

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path

        # Load SmolVLM processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Load model with appropriate dtype
        dtype = self._get_dtype(self.dtype)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            **config.extra_args.get("model_kwargs", {})
        )

        self.model.eval()
        print(f"[LOGS] SmolVLM loaded on {self.device}")

    def process_image(self, image):
        """Process PIL image for SmolVLM model."""
        return image  # SmolVLM processor handles image preprocessing

    def process_image_path(self, img_path):
        """Load image from file path."""
        image = Image.open(img_path).convert("RGB")
        return image

    def generate(self, image, prompt: str | None = None):
        """Generate text response from image and prompt."""
        if prompt is None:
            prompt = "Describe the content of this image in detail."

        # Create messages format (SmolVLM uses chat template)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Prepare inputs
        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with model
        # with torch.no_grad():
        #     output_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=self.config.max_new_tokens,
        #         temperature=self.config.temperature,
        #         top_p=self.config.top_p,
        #         do_sample=self.config.temperature > 0,
        #     )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip input)
        generated_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()

        return generated_text

    def predict(self, img_path, prompt=None):
        """Main inference method called by evaluation framework."""
        image = self.process_image_path(img_path)
        return self.generate(image, prompt)