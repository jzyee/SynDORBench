"""
requires:
- transformers >= 4.45
- microsoft/Phi-4-multimodal-instruct

This class integrates Phi-4 Multimodal Instruct (text+image) into the SynDORBench framework.

example from: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_inference_phi4mm.py
"""

from eval.models.base import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import requests


class Phi4(BaseModel):
    """
    Wrapper for Microsoft's Phi-4-Multimodal-Instruct model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path
        self.max_new_tokens = config.max_new_tokens

        # ------------------------------------------------------------
        # Load model + processor
        # ------------------------------------------------------------
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map=self.device_map,
        )

        # Optional generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception:
            self.generation_config = GenerationConfig()

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None):
        """
        Run inference on an image and text prompt using Phi-4 multimodal format.

        Parameters
        ----------
        img_path : str
            Local path or URL to image.
        prompt : str, optional
            User question or instruction.

        Returns
        -------
        str
            Model's textual response.
        """

        if prompt is None:
            prompt = "What is shown in this image?"

        # Prepare the Phi-style multimodal prompt
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        full_prompt = f"{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}"

        # Load image (URL or local path)
        if img_path.startswith("http://") or img_path.startswith("https://"):
            image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        # Tokenize
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.inference_mode():

            if not hasattr(self.model.config, "num_logits_to_keep") or self.model.config.num_logits_to_keep is None:
                self.model.config.num_logits_to_keep = 1
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.generation_config,
                do_sample=False,
                num_logits_to_keep=1,
            )

        # Strip the input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        # Decode
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response

    @torch.inference_mode()
    def score_options(self, img_path, prompt, option_letters):
        """A-only MCQ scoring: mirrors generate()'s Phi-4 prompt/image
        construction exactly, but swaps model.generate() for a direct
        forward pass and reads logits restricted to option_letters at the
        answer position (single forward pass, softmax over just those
        tokens, argmax)."""
        from eval.models.mcq_utils import letter_token_ids

        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        full_prompt = f"{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}"

        if str(img_path).startswith("http://") or str(img_path).startswith("https://"):
            image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        option_ids = letter_token_ids(self.processor.tokenizer, option_letters)

        outputs = self.model(**inputs)
        last_logits = outputs.logits[:, -1, :].squeeze(0)
        option_logits = last_logits[option_ids].float()
        probs = torch.softmax(option_logits, dim=-1).tolist()
        pred_letter = option_letters[int(torch.argmax(option_logits).item())]
        return pred_letter, probs

    # ------------------------------------------------------------
    # Predict (SynDORBench entry point)
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None):
        """
        Unified SynDORBench predict method.
        """
        return self.generate(str(img_path), prompt)
