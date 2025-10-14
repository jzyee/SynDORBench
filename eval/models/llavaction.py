
'''
requires:
- transformers
- llavaction



'''
from eval.models.base import BaseModel
from llavaction.model.builder import load_pretrained_model
import torch
from llavaction.conversation import conv_templates, SeparatorStyle
from llavaction.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llavaction.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
import copy
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel


class LLavaAction(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        # Initialize LLavaAction specific components here
        # e.g., load model, tokenizer, etc.

        self.model_path = config.model_path

        tokenizer, model, img_processor, max_length = load_pretrained_model(self.model_path, 
                                                                            None, 
                                                                            self.model_name, 
                                                                            torch_dtype=self.dtype, 
                                                                            device_map=self.device_map,
                                                                            # vision_tower=self.config.vision_tower,
                                                                            quantization_config=self.config.extra_args.get("quantization_config", None)
                                                                            )
        # tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        # img_processor = AutoImageProcessor.from_pretrained(self.model_path)
        # model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.config.dtype).to(self.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = img_processor

    def process_image(self, image):
        # Implement image processing logic specific to LLavaAction
        return self.image_processor.preprocess(image, return_tensors="pt")['pixel_values'].cuda().to(torch.bfloat16)

    def process_image_path(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return self.process_image(image)

    def generate(self, image, prompt: str | None = None):
        conv_template = "qwen_2"

        if prompt is None:

            

            time_instruction = f"The video lasts for {0:.2f} seconds, and 1 frame from it."

            perspective_prompt = "You are seeing this video from a surveillance camera view and you are the surveillance camera. What action is the person performing?"
            task_prompt = "Describe in details what you see from the video frames."

            base_text = f"\n{time_instruction}\n{perspective_prompt} {task_prompt}"

        else:
            base_text = prompt.strip()

        if DEFAULT_IMAGE_TOKEN in base_text:
            question = base_text
        else:
            question = DEFAULT_IMAGE_TOKEN + base_text

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        cont = self.model.generate(
            input_ids,
            images=image,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        return text_outputs
    
    def predict(self, img_path, prompt=None):
        image = self.process_image_path(img_path)
        outputs = self.generate(image, prompt)
        return outputs