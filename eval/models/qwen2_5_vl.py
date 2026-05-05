"""
requires:
- transformers
- Qwen/Qwen2.5-VL-7B-Instruct
- qwen_vl_utils (from Qwen-VL repository)

This class integrates Qwen2.5-VL-7B-Instruct into the SynDORBench evaluation framework.
"""

from eval.models.base import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch


class Qwen2_5_VL(BaseModel):
    """
    Wrapper for the Qwen2.5-VL-7B-Instruct model, providing a standardized
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2",  # enable efficient attention
        ).eval()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on an image (or video) with an optional text prompt.

        Parameters
        ----------
        img_path : str
            Path or URL of the image or video.
        prompt : str, optional
            Textual instruction for the model.

        Returns
        -------
        str
            Generated text output (decoded, trimmed, and cleaned).
        """

        if prompt is None:
            prompt = "Describe this image in detail."

        # ------------------------------------------------------------
        # Build message for multimodal chat
        # ------------------------------------------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # ------------------------------------------------------------
        # Prepare inputs
        # ------------------------------------------------------------
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # ------------------------------------------------------------
        # Run generation
        # ------------------------------------------------------------
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        # Trim off input tokens from output
        generated_tokens = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode model output
        decoded_output = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        return decoded_output

    # ------------------------------------------------------------
    # Predict (standardized entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        Standardized SynDORBench entry point for model inference.
        """
        return self.generate(img_path, prompt)
