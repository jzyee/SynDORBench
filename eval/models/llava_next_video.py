'''
requires:
- transformers
- llava-hf/LLaVA-NeXT-Video-7B-hf

This class integrates the LLaVA-NeXT-Video model into the SynDORBench evaluation framework.
It follows the same BaseModel structure used for LLaVAction, ensuring consistency.
'''

from eval.models.base import BaseModel
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from pathlib import Path


class LlavaNextVideo(BaseModel):
    """
    Wrapper class for the LLaVA-NeXT-Video model, built to handle video-based
    visual-language tasks within the SynDORBench evaluation framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path

        # ------------------------------------------------------------
        # Load model and processor
        # ------------------------------------------------------------
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self.model.eval()

    # ------------------------------------------------------------
    # Video Processing
    # ------------------------------------------------------------
    def process_video(self, video_path: str):
        """
        Process a video into model-compatible tensor format.

        Parameters
        ----------
        video_path : str
            Path to the input video file.

        Returns
        -------
        dict
            Dictionary of processed video tensors ready for model input.
        """
        video_path = str(Path(video_path))
        inputs = self.processor(video_path, return_tensors="pt").to(self.model.device)
        return inputs

    # ------------------------------------------------------------
    # Text Generation (Response Extraction)
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on a given video with an optional prompt.

        Returns
        -------
        str
            Model’s response text (excluding prompt).
        """

        if prompt is None:
            # Default system prompt similar to LLavaAction
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what action is being performed?"},
                        {"type": "image", "path": str(img_path)},
                        ],
                },
            ]

        else:
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "path": str(img_path)},
                        ],
                },
            ]

        # Preprocess video and prompt
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        # remove prompt from output
        response_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        # Decode full output and strip prompt from it
        response = self.processor.batch_decode(response_tokens, skip_special_tokens=True)[0].strip()

        return response

    # ------------------------------------------------------------
    # Predict (Entry point for benchmarking)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        High-level wrapper for standardized prediction interface.

        Returns
        -------
        str
            Clean model response text.
        """
        return self.generate(img_path, prompt)
