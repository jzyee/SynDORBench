"""
requires:
- transformers
- torch
- pillow
- git+https://github.com/ictnlp/LLaVA-Mini.git
"""

from eval.models.base import BaseModel
import torch
from PIL import Image
import copy


class LLavaMini(BaseModel):
    """Wrapper for ICT-NLP's LLaVA-Mini model compatible with SynDORBench eval framework."""

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.model_path

        try:
            # ✅ Use LLaVA-Mini-specific modules
            from llavamini.model.builder import load_pretrained_model
            from llavamini.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llavamini.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llavamini.conversation import conv_templates

            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.conv_templates = conv_templates
            self.tokenizer_image_token = tokenizer_image_token

            # Load pretrained LLaVA-Mini model
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                self.model_path,
                None,
                get_model_name_from_path(self.model_path),
                torch_dtype=self.dtype,
                device_map=self.device_map,
                **config.extra_args.get("model_kwargs", {})
            )

            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self.max_length = max_length

        except ImportError:
            # Fallback to standard transformers if llava-mini not installed
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
            print("⚠️ LLaVA-Mini package not found — using transformers fallback")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.image_processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )

            self.IMAGE_TOKEN_INDEX = -200
            self.DEFAULT_IMAGE_TOKEN = "<image>"
            self.conv_templates = None

        self.model.eval()

    # -----------------------------
    # Image preprocessing
    # -----------------------------
    def process_image(self, image):
        """Convert PIL image to tensor for LLaVA-Mini."""
        if hasattr(self.image_processor, "preprocess"):
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ]
        else:
            image_tensor = self.image_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ]
        return image_tensor.to(self.device, dtype=self.dtype)

    def process_image_path(self, img_path):
        """Load and preprocess image from file path."""
        image = Image.open(img_path).convert("RGB")
        return self.process_image(image)

    # -----------------------------
    # Generation
    # -----------------------------
    def generate(self, image, prompt: str | None = None):
        """Generate output from image and prompt using LLaVA-Mini."""
        conv_template = "vicuna_v1"

        if prompt is None:
            prompt = "Describe the content of the image in detail."

        # Add image token if not present
        if self.DEFAULT_IMAGE_TOKEN not in prompt:
            question = f"{self.DEFAULT_IMAGE_TOKEN}\n{prompt}"
        else:
            question = prompt

        if self.conv_templates is not None:
            conv = copy.deepcopy(self.conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            input_ids = self.tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
        else:
            input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(
                self.device
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text

    # -----------------------------
    # Inference interface
    # -----------------------------
    def predict(self, img_path, prompt=None):
        image = self.process_image_path(img_path)
        return self.generate(image, prompt)
