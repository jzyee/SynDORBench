"""
requires:
- openai >= 1.0.0
- pillow
- environment variable OPENAI_API_KEY set

This class integrates OpenAI's GPT-5-Mini (or GPT-4o-Mini)
multimodal API into the SynDORBench evaluation framework.
It resizes every input image to 512×512 before encoding and
includes automatic retry/wait when hitting API rate limits.
"""

from eval.models.base import BaseModel
from openai import OpenAI, APIError, RateLimitError, InternalServerError
from PIL import Image
import base64
import mimetypes
import os
import io
import time


class OpenAI4oMini(BaseModel):
    """
    Wrapper for OpenAI GPT-5-Mini (or GPT-4o-Mini) via API.
    Provides a standardized predict() interface for SynDORBench.
    Automatically handles rate-limit retries with exponential backoff.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_path = getattr(config, "model_path", "gpt-5-mini")
        self.max_output_tokens = getattr(config, "max_new_tokens", 256)
        self.client = OpenAI()

        # Fixed resize dimensions (square)
        self.target_size = getattr(config, "target_size", (512, 512))

        # Retry parameters
        self.max_retries = getattr(config, "max_retries", 5)
        self.initial_backoff = getattr(config, "initial_backoff", 10)  # seconds

    # ------------------------------------------------------------
    # Helper: Resize to 512×512 and encode local image to base64
    # ------------------------------------------------------------
    def encode_image(self, image_path: str) -> str:
        """
        Loads an image, resizes it to 512×512, and returns
        it as a base64 data URI string.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"

        # --- Load, convert, resize ---
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

            # --- Encode to base64 ---
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:{mime_type};base64,{b64}"

    # ------------------------------------------------------------
    # Core generation logic (with rate-limit handling)
    # ------------------------------------------------------------
    def generate(self, img_path: str, prompt: str | None = None) -> str:
        """
        Perform multimodal inference with GPT-5-Mini or GPT-4o-Mini.
        Includes exponential-backoff retry when hitting rate limits.
        """
        if prompt is None:
            prompt = "Describe this image in detail."

        image_data_uri = self.encode_image(img_path)
        backoff = self.initial_backoff

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model_path,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": image_data_uri},
                            ],
                        }
                    ],
                    max_output_tokens=self.max_output_tokens,
                )

                # --- Extract output text ---
                try:
                    output_text = response.output_text.strip()
                except AttributeError:
                    output_text = (
                        response.choices[0].message.get("content", "").strip()
                        if hasattr(response, "choices")
                        else ""
                    )

                if hasattr(response, "usage"):
                    print(f"[OpenAI] Tokens used: {response.usage}")

                return output_text or "[No response generated]"

            except RateLimitError as e:
                print(f"⚠️ Rate limit hit (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    print(f"⏳ Waiting {backoff} s before retry...")
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                else:
                    print("❌ Maximum retries reached. Aborting.")
                    raise

            except (APIError, InternalServerError) as e:
                # Retry transient API or server errors
                print(f"⚠️ API/server error (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    print(f"⏳ Waiting {backoff} s before retry...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print("❌ Maximum retries reached. Aborting.")
                    raise

            except Exception as e:
                print(f"❌ Unexpected error during inference: {e}")
                raise

    # ------------------------------------------------------------
    # Standard SynDORBench predict() interface
    # ------------------------------------------------------------
    def predict(self, img_path: str, prompt: str | None = None) -> str:
        return self.generate(img_path, prompt)


# ============================================================
# Standalone connectivity test
# ============================================================
if __name__ == "__main__":
    from eval.models.base import ModelConfig

    print("🔍 Checking environment variable...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set it with:")
        print('   export OPENAI_API_KEY="sk-xxxx"')
        exit(1)
    else:
        print("✅ OPENAI_API_KEY detected.")

    cfg = ModelConfig.from_dict({
        "model_name": "gpt-4o-mini",
        "model_path": "gpt-4o-mini",
        "max_new_tokens": 512,
        "target_size": (512, 512),
        "max_retries": 5,
        "initial_backoff": 10,
    })

    model = OpenAI4oMini(cfg)

    # path to an image
    img_path = (
        ""
    )
    print(f"\n🧠 Running {cfg.model_name} test inference on: {img_path}")
    result = model.predict(img_path, "Describe the image precisely.")
    print("\n🗒️  Model Output:")
    print(result)
