"""Read Urdu text from line images or whole pages."""
from __future__ import annotations

import torch
from PIL import Image

from uocr.config import BASE_MODEL, DEFAULT_MAX_SIDE
from uocr.modeling import load_image, load_model, load_processor
from uocr.pages import segment_lines
from uocr.prompts import INSTRUCTION, build_messages, clean_output


class UrduReader:
    def __init__(self, base_model: str = BASE_MODEL, adapter_path: str | None = None,
                 max_side: int = DEFAULT_MAX_SIDE, quantize_4bit: bool = False,
                 instruction: str = INSTRUCTION):
        self.processor = load_processor(base_model)
        self.processor.tokenizer.padding_side = "left"
        self.model = load_model(base_model, adapter_path, quantize_4bit=quantize_4bit)
        # Do NOT merge_and_unload() here: it is ~1.7x faster, but folding LoRA deltas into bf16
        # weights changed greedy outputs (1 of 4 test lines identical, up to 33% CER apart).
        self.model.eval()
        self.max_side = max_side
        self.instruction = instruction

    @torch.inference_mode()
    def read_lines(self, images, max_new_tokens: int = 256, batch_size: int = 8) -> list[str]:
        out = []
        for i in range(0, len(images), batch_size):
            imgs = [load_image(im, self.max_side) for im in images[i:i + batch_size]]
            texts = [self.processor.apply_chat_template(build_messages(im, instruction=self.instruction), tokenize=False,
                                                        add_generation_prompt=True) for im in imgs]
            enc = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.model.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05)
            out += [clean_output(t) for t in self.processor.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out

    def read_page(self, img: Image.Image) -> tuple[str, list[Image.Image]]:
        lines = segment_lines(img)
        return "\n".join(self.read_lines(lines)), lines
