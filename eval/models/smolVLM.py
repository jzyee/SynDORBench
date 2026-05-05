"""
requires:
- transformers
- HuggingFaceTB/SmolVLM2-2.2B-Instruct

This class integrates SmolVLM2-2.2B-Instruct into the SynDORBench evaluation framework.
"""

from eval.models.base import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


class SmolVLM(BaseModel):
    """
    Wrapper for the SmolVLM2-2.2B-Instruct model, providing a standardized
    predict() interface for integration into the SynDORBench evaluation pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens
        # ------------------------------------------------------------
        # Load processor and model
        # ------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device_map
        ).eval()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on an image with an optional text prompt.

        Parameters
        ----------
        img_path : str
            Path or URL of the image.
        prompt : str, optional
            The textual instruction.

        Returns
        -------
        str
            Model response text (excluding special tokens).
        """

        if prompt is None:
            prompt = "Describe this image in detail."

        # Build message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Tokenize and prepare input tensors
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Run generation
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        generated_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]

        # Decode response text
        decoded_output = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        return decoded_output

    # ------------------------------------------------------------
    # Predict (standardized entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        return self.generate(img_path, prompt)
