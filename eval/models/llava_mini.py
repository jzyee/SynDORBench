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

        # ============================================================
        # Device & dtype setup
        # ============================================================
        device_str = getattr(config, "device", "cuda:0")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        dtype_str = getattr(config, "dtype", "bfloat16")
        # Keep user's preferred dtype for loading, but at runtime we will
        # ALWAYS match tensors to the model's actual parameter dtype.
        self.dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str

        # Single-GPU mode recommended for stability
        self.device_map = getattr(config, "device_map", None)  # should be None

        try:
            from llavamini.model.builder import load_pretrained_model
            from llavamini.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llavamini.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llavamini.conversation import conv_templates

            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.conv_templates = conv_templates
            self.tokenizer_image_token = tokenizer_image_token

            tokenizer, model, image_processor, max_length = load_pretrained_model(
                self.model_path,
                None,
                get_model_name_from_path(self.model_path),
                torch_dtype=self.dtype,
                device_map=self.device_map,   # None → single GPU
                **config.extra_args.get("model_kwargs", {}),
            )
            self.tokenizer = tokenizer
            self.model = model.to(self.device)
            self.image_processor = image_processor
            self.max_length = max_length

        except ImportError:
            # Fallback to standard transformers
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
            ).to(self.device)

            self.IMAGE_TOKEN_INDEX = -200
            self.DEFAULT_IMAGE_TOKEN = "<image>"
            self.conv_templates = None

        self.model.eval()

    # ============================================================
    # Utilities
    # ============================================================
    def _model_dtype(self) -> torch.dtype:
        """Return the model's actual parameter dtype (ground-truth for casting)."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return getattr(self.model, "dtype", self.dtype)

    def _model_device(self) -> torch.device:
        """Return the model's device (for aligning inputs)."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    # ============================================================
    # Image preprocessing (single image → 1-frame video)
    # ============================================================
    def process_image(self, image):
        """Convert a PIL image into a (B, 1, C, H, W) tensor for LLaVA-Mini."""
        if hasattr(self.image_processor, "preprocess"):
            image_tensor = self.image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"]
        else:
            image_tensor = self.image_processor(
                images=image, return_tensors="pt"
            )["pixel_values"]

        # Ensure correct shape: (B, 1, C, H, W)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(1)

        # ❗ CRITICAL: cast to the model's *actual* dtype & device
        target_dtype = self._model_dtype()
        target_device = self._model_device()
        return image_tensor.to(device=target_device, dtype=target_dtype, non_blocking=True)

    def process_image_path(self, img_path):
        """Load and preprocess image from file path."""
        image = Image.open(img_path).convert("RGB")
        return self.process_image(image)

    # ============================================================
    # Generation
    # ============================================================
    def generate(self, image, prompt: str | None = None):
        """Generate output from image and prompt using LLaVA-Mini."""
        conv_template = "vicuna_v1"
        if prompt is None:
            prompt = "Describe the content of the image in detail."

        # Ensure image token is included
        question = (
            f"{self.DEFAULT_IMAGE_TOKEN}\n{prompt}"
            if self.DEFAULT_IMAGE_TOKEN not in prompt
            else prompt
        )

        # =============================
        # Construct conversation prompt
        # =============================
        if self.conv_templates is not None:
            conv = copy.deepcopy(self.conv_templates[conv_template])

            # --- Vicuna-style conversation object ---
            if hasattr(conv, "append_message"):
                if isinstance(conv.messages, tuple):
                    conv.messages = list(conv.messages)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_text = conv.get_prompt()

            # --- Qwen/legacy-style iterable template ---
            else:
                conv = list(conv)
                conv.append(("user", question))
                conv.append(("assistant", None))
                prompt_text = "\n".join(f"{r}: {m}" for r, m in conv if m)

            input_ids = self.tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0)

        else:
            input_ids = self.tokenizer(question, return_tensors="pt").input_ids

        # Align input_ids to the model's embedding device and correct dtype
        embed_device = self._model_device()
        input_ids = input_ids.to(device=embed_device, dtype=torch.long)

        # =============================
        # Generation
        # =============================
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image,  # (B, 1, C, H, W), already on correct device/dtype
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        # =============================
        # Decode text
        # =============================
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text

    # ============================================================
    # Inference interface
    # ============================================================
    def predict(self, img_path, prompt=None):
        """Unified predict interface for SynDORBench."""
        image = self.process_image_path(img_path)
        return self.generate(image, prompt)
