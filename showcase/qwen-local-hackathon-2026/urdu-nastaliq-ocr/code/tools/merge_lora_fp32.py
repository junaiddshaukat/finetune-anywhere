"""Merge a LoRA adapter into Qwen3-VL in fp32 on the CPU, for GGUF conversion (llama.cpp convert_hf_to_gguf.py).

    CUDA_VISIBLE_DEVICES="" python tools/merge_lora_fp32.py Qwen/Qwen3-VL-2B-Instruct outputs/<adapter> outputs/<merged-dir>

fp32 + CPU so the merge is exact (bf16 merging changed greedy outputs, see uocr/inference.py). No uocr imports.
"""
import sys, torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

base, adapter, out = sys.argv[1:4]
m = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.float32, device_map="cpu")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out)
AutoProcessor.from_pretrained(base).save_pretrained(out)
print("MERGED", out)
