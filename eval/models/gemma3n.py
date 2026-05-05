"""
requires:
- transformers
- google/gemma-3n-e4b-it

This class integrates Gemma 3 N into the SynDORBench evaluation framework.
"""

from eval.models.base import BaseModel
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch


class Gemma3n(BaseModel):
    """
    Wrapper class for the Gemma 3 N model, aligned with SynDORBench’s evaluation framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens
        # ------------------------------------------------------------
        # Load processor and model
        # ------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=self.dtype,
        ).eval()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on an image and return only the model’s response text.

        Parameters
        ----------
        img_path : str
            Path or URL of the image.
        prompt : str, optional
            The text prompt for the model.

        Returns
        -------
        str
            Model response text (without prompt or special tokens).
        """

        if prompt is None:
            prompt = "Describe what is happening in this image."

        # Build a multimodal chat-style message
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a surveillance system."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
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

        # Generate output
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            # Keep only the newly generated tokens
            generation = generation[0][input_len:]

        # Decode and return clean text
        decoded = self.processor.decode(generation, skip_special_tokens=True).strip()
        return decoded

    # ------------------------------------------------------------
    # Predict (standardized SynDORBench entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        High-level standardized prediction interface.

        Returns
        -------
        str
            Clean model response text.
        """
        return self.generate(img_path, prompt)
