"""
requires:
- transformers >= 4.45
- allenai/Molmo-7B-D-0924

This class integrates AllenAI’s Molmo-7B-D-0924 multimodal model into the SynDORBench framework.


https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/48
"""

from eval.models.base import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import requests


class Molmo(BaseModel):
    """
    Wrapper class for AllenAI’s Molmo-7B-D-0924 multimodal model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens

        # ------------------------------------------------------------
        # Load model and processor
        # ------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            # torch_dtype=self.dtype,
            device_map=self.device_map,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            # torch_dtype=self.dtype,
            device_map=self.device_map,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception:
            self.generation_config = GenerationConfig()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Generate a textual description or answer for an input image and prompt.

        Parameters
        ----------
        img_path : str
            Path or URL to the input image.
        prompt : str, optional
            Textual query to describe or interpret the image.

        Returns
        -------
        str
            Model-generated response text.
        """

        if prompt is None:
            prompt = "Describe this image."

        # Load image (support both URL and local path)
        if img_path.startswith("http://") or img_path.startswith("https://"):
            image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        # Process multimodal input
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}


        # print(f"input type: {type(inputs)}")

        # Generate
        with torch.inference_mode():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    stop_strings="<|endoftext|>",
                    use_cache=True,
                ),
                tokenizer=self.processor.tokenizer,
            )

        

        # Decode generated portion only
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return generated_text

    # ------------------------------------------------------------
    # Predict (SynDORBench entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        Standardized interface for SynDORBench evaluation.
        """
        # print(str(img_path))
        return self.generate(str(img_path), prompt)
