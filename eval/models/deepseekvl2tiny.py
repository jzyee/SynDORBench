
"""
DeepSeek-VL2-Tiny model wrapper for SynDORBench evaluation framework.

Requirements:
- transformers>=4.37.0
- torch>=2.0
- pillow
- accelerate
- pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
- pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
"""

from .base import BaseModel
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVl2Tiny(BaseModel):
    """Wrapper for DeepSeek-VL2-Tiny (3B) vision-language model."""

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path

        print(f"[LOGS] Loading DeepSeek-VL2-Tiny from {self.model_path}...")

        print("[LOGS] Loading processor...")
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            self.model_path
        )

        print("[LOGS] Loading model...")
        dtype = self._get_dtype(self.dtype)

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        self.vl_gpt = self.vl_gpt.to(self.device).eval()
        self.model = self.vl_gpt  # For compatibility

        print(f"[LOGS] DeepSeek-VL2-Tiny loaded successfully on {self.device}")

    def process_image(self, image):
        """Process PIL image for DeepSeek model."""
        # DeepSeek uses processor for image preprocessing
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        return inputs["pixel_values"].to(self.device, dtype=self._get_dtype(self.dtype))

    def process_image_path(self, img_path):
        """Load and preprocess image from file path."""
        image = Image.open(img_path).convert("RGB")
        return self.process_image(image)

    def generate(self, img_path, prompt: str | None = None):
        """Generate text response from image path and prompt."""
        if prompt is None:
            prompt = "Describe the content of this image in detail."

        # DeepSeek format
        conversation = [
            {
                "role": "user",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [img_path],
            },
            {"role": "assistant", "content": ""},
        ]

        pil_images = load_pil_images(conversation)

        processed = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        )

        processed = processed.to(self.vl_gpt.device)

        # Access attributes directly (not dict-like access)
        input_ids = processed.input_ids
        attention_mask = getattr(processed, "attention_mask", None)
        images = getattr(processed, "images", None)
        images_seq_mask = getattr(processed, "images_seq_mask", None)
        images_spatial_crop = getattr(processed, "images_spatial_crop", None)

        input_length = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            try:
                outputs = self.vl_gpt.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    do_sample=self.config.temperature > 0,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p if self.config.temperature > 0 else None,
                )
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape if input_ids is not None else None}")
                print(f"[DEBUG] images shape: {images.shape if images is not None else None}")
                raise

        # Decode ONLY the generated tokens
        generated_tokens = outputs[0, input_length:]  # Skip the input tokens
        generated_text = self.processor.tokenizer.decode(
            generated_tokens.cpu().tolist(),
            skip_special_tokens=True
        ).strip()

        return generated_text

    def predict(self, img_path, prompt=None):
        """Main inference method called by evaluation framework."""
        raw_output = self.generate(img_path, prompt)

        # Post-process to extract binary answer if this is a binary classification task
        if prompt and ("1 or 0" in prompt.lower() or "0 or 1" in prompt.lower()):
            # Remove and clean common prefixes for the output
            cleaned = raw_output.strip()

            # Remove image placeholder if present
            if "<image_placeholder>" in cleaned:
                cleaned = cleaned.replace("<image_placeholder>", "").strip()

            # Extract the first line or first character thats 0 or 1
            lines = cleaned.split('\n')
            for line in lines:
                line = line.strip()
                if line in ['0', '1']:
                    return line
                # Check if line starts with 0 or 1
                if line and line[0] in ['0', '1']:
                    return line[0]

            # Fallback: search for first 0 or 1 in the entire output
            for char in cleaned:
                if char in ['0', '1']:
                    return char

        return raw_output