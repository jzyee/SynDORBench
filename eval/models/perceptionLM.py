"""
requires:
- transformers >= 4.44
- facebook/Perception-LM-3B

This class integrates Perception-LM into the SynDORBench evaluation framework.
"""

from eval.models.base import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from huggingface_hub import hf_hub_download


class PerceptionLM(BaseModel):
    """
    Wrapper for Facebook’s Perception-LM model.
    Designed to align with SynDORBench’s multimodal evaluation structure.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens
        self.device = self.device

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=config.device_map,
        ).to(self.device).eval()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on an image and return model output text.

        Parameters
        ----------
        img_path : str
            Path to image (local or HF dataset path).
        prompt : str, optional
            Text query for the image.

        Returns
        -------
        str
            Decoded text output.
        """

        if prompt is None:
            prompt = "Describe the content of this image."

        # Build conversation input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Tokenize and prepare for model
        inputs = self.processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Trim input tokens
        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_length:]

        # Decode
        decoded_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return decoded_output.strip()

    # ------------------------------------------------------------
    # Predict (SynDORBench standardized entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        Standard prediction interface used by SynDORBench.

        Returns
        -------
        str
            Model text response.
        """
        return self.generate(img_path, prompt)
