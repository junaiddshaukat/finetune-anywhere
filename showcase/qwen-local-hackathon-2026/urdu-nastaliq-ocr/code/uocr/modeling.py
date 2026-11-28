"""Load Qwen3-VL (+ optional LoRA adapter) for training or inference."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from uocr.config import HF_TOKEN

LM_LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

# Never upscale tiny line images below this height (Nastaliq dots/ligatures need pixels).
MIN_HEIGHT = 48


def load_image(path_or_img, max_side: int) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    if img.height < MIN_HEIGHT:
        scale = MIN_HEIGHT / img.height
        img = img.resize((max(1, int(img.width * scale)), MIN_HEIGHT), Image.BICUBIC)
    return img


def load_processor(model_id: str):
    return AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)


def load_model(model_id: str, adapter_path: str | None = None, quantize_4bit: bool = False,
               device_map="auto"):
    """Loads Qwen3-VL by default; Qwen2-VL checkpoints (e.g. the Qaari baseline) are detected by name."""
    model_cls = Qwen3VLForConditionalGeneration
    if "qwen2-vl" in model_id.lower():
        from transformers import Qwen2VLForConditionalGeneration
        model_cls = Qwen2VLForConditionalGeneration
    kwargs = {"dtype": torch.bfloat16, "device_map": device_map, "token": HF_TOKEN}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = model_cls.from_pretrained(model_id, **kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    return model
