"""
requires:
- transformers
- google/gemma-3-4b-it

This class integrates Gemma 3 (4B-IT) into the SynDORBench evaluation framework.
"""

from eval.models.base import BaseModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch


class Gemma3(BaseModel):
    """
    Wrapper class for the Gemma 3 (4B-IT) model, aligned with SynDORBench’s evaluation framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens

        # ------------------------------------------------------------
        # Load processor and model
        # ------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            padding_side="left"
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=self.dtype,
            attn_implementation="flash_attention_2",
        ).eval()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str | None = None, prompt: str | None = None):
        """
        Run inference on an image (or text-only if no image) and return the model’s response text.

        Parameters
        ----------
        img_path : str, optional
            Path or URL of the image.
        prompt : str, optional
            Text prompt for the model.

        Returns
        -------
        str
            Clean decoded response text.
        """

        if prompt is None:
            prompt = "Describe what is shown in this image."

        # Build a multimodal chat message
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a captioning system."}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": str(img_path)}] if img_path else []
                ) + [{"type": "text", "text": prompt}],
            },
        ]

        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                cache_implementation="static",
            )
            generation = generation[0][input_len:]

        # Decode output
        decoded = self.processor.decode(generation, skip_special_tokens=True).strip()
        return decoded

    # ------------------------------------------------------------
    # Predict (standardized SynDORBench entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str | None = None, prompt: str | None = None):
        """
        High-level standardized prediction interface for SynDORBench.

        Returns
        -------
        str
            Model response text.
        """
        return self.generate(img_path, prompt)
