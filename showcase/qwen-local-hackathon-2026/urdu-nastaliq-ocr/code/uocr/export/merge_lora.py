"""Merge the LoRA adapter into the base model (for sharing, or GGUF conversion for phones).

Usage:
    python -m uocr.export.merge_lora --adapter outputs/qwen3vl2b-urdu-ocr-lora --out outputs/merged-2b
    # Qaari 0.1 as a standalone model (base for continued fine-tuning):
    python -m uocr.export.merge_lora --qaari --out outputs/qaari-qwen2-vl-merged

For llama.cpp (offline phone/laptop), from a llama.cpp checkout:
    python convert_hf_to_gguf.py <merged_dir> --outfile urdu-ocr-2b-f16.gguf --outtype f16
    python convert_hf_to_gguf.py <merged_dir> --mmproj --outfile urdu-ocr-2b-mmproj-f16.gguf --outtype f16
    ./llama-quantize urdu-ocr-2b-f16.gguf urdu-ocr-2b-Q4_K_M.gguf Q4_K_M
Check the current llama.cpp multimodal docs (flags change), and re-run the eval on the
quantized model: 4-bit can hurt OCR accuracy more than chat quality.
"""
from __future__ import annotations

import argparse

from uocr.config import BASE_MODEL, ROOT
from uocr.modeling import load_model, load_processor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--qaari", action="store_true", help="merge the Qaari 0.1 adapter into Qwen2-VL-2B")
    args = ap.parse_args()
    if args.qaari:  # Qwen2-VL detected by name in load_model; keep "qwen2-vl" in --out for the same reason
        from uocr.eval.evaluate import QAARI_BASE, qaari_adapter_dir
        args.base_model, args.adapter = QAARI_BASE, qaari_adapter_dir()
    if not args.adapter:
        ap.error("--adapter is required (or --qaari)")
    model = load_model(args.base_model, str(ROOT / args.adapter), device_map="cpu").merge_and_unload()
    model.save_pretrained(ROOT / args.out)
    load_processor(args.base_model).save_pretrained(ROOT / args.out)
    print(f"Merged model -> {ROOT / args.out}")


if __name__ == "__main__":
    main()
