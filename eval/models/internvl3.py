'''
requires:
- transformers
- torch
- OpenGVLab/InternVL3-1B-hf

This class integrates the InternVL3 model into the SynDORBench evaluation framework.
It follows the same BaseModel interface as LLavaAction, ensuring consistent evaluation.
'''

from eval.models.base import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path


class InternVL3(BaseModel):
    """
    Wrapper class for the OpenGVLab InternVL3 model.
    Handles image + text input and returns the model’s pure response.
    """

    def __init__(self, config):
        super().__init__(config)

        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens
        # ------------------------------------------------------------
        # Load model and processor
        # ------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=self.dtype,
        )

        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------
    # Image Processing
    # ------------------------------------------------------------
    def process_image_path(self, img_path: str | Path):
        """
        Ensure the image path is a string and accessible for model input.
        """
        return str(Path(img_path))

    # ------------------------------------------------------------
    # Text Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str | Path, prompt: str | None = None):
        """
        Run inference on a single image with an optional text prompt.

        Parameters
        ----------
        img_path : str | Path
            Path to the image file.
        prompt : str, optional
            The text prompt or question.

        Returns
        -------
        str
            Model’s generated text (excluding the prompt).
        """

        img_path = self.process_image_path(img_path)

        # Construct conversation message
        if prompt is None:
            text_prompt = "Describe the image in full detail."
        else:
            text_prompt = prompt.strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Tokenize + format
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Inference
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Extract only model response tokens (omit prompt)
        response_tokens = generate_ids[:, inputs["input_ids"].shape[1]:]
        decoded_output = self.processor.decode(
            response_tokens[0],
            skip_special_tokens=True
        ).strip()

        return decoded_output

    # ------------------------------------------------------------
    # Predict (entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str | Path, prompt: str | None = None):
        """
        Standardized prediction interface for SynDORBench.
        """
        return self.generate(img_path, prompt)
