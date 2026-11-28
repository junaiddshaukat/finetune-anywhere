"""Evaluate a quantized GGUF build through llama-server, on the same test items and metrics as
uocr.eval.evaluate, so its numbers drop straight into `--compare`.

    llama-server -m outputs/gguf/urdu-ocr-2b-Q4_K_M.gguf --mmproj outputs/gguf/urdu-ocr-2b-mmproj-f16.gguf \
        --port 8090 -c 32768 -np 4 -ngl 99 --jinja
    python tools/eval_gguf.py --name gguf_q4 --url http://127.0.0.1:8090 --workers 4
    python -m uocr.eval.evaluate --compare finetuned gguf_q8 gguf_q4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uocr.config import DEFAULT_MAX_SIDE  # noqa: E402
from uocr.eval.evaluate import EVAL_DIR, PROCESSED_DIR, load_test, metrics  # noqa: E402
from uocr.pages import prepare_for_reading  # noqa: E402
from uocr.prompts import INSTRUCTION, clean_output  # noqa: E402


def read_one(url: str, row: dict, max_new_tokens: int, auto_scale: bool = False, dry: bool = False) -> dict:
    buf = io.BytesIO()
    prepare_for_reading(PROCESSED_DIR / row["image"], DEFAULT_MAX_SIDE, auto_scale).save(buf, format="PNG")
    body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": INSTRUCTION}]}],
            "temperature": 0, "max_tokens": max_new_tokens, "repeat_penalty": 1.05}  # as in UrduReader
    if dry:   # llama.cpp DRY sampler: penalises continuing a sequence that already occurred (anti-looping)
        body.update(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=3, dry_penalty_last_n=2048)
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = clean_output(json.load(r)["choices"][0]["message"]["content"])
    except OSError as e:
        print(f"  request failed on {row['id']}: {e}")
        out = ""
    return {**row, "pred": out, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--dry", action="store_true", help="enable llama.cpp's DRY anti-repetition sampler")
    ap.add_argument("--auto-scale", action="store_true",
                    help="enlarge densely packed multi-line images (uocr.pages.scale_for_reading)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (match llama-server -np)")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.url}/props", timeout=10) as r:
        model = Path(json.load(r).get("model_path", "?")).name
    rows = load_test(args.per_source or None, args.sources)
    print(f"Evaluating '{args.name}' ({model}) on {len(rows)} test lines, {args.workers} worker(s)")
    preds, t0 = [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, p in enumerate(pool.map(lambda r: read_one(args.url, r, args.max_new_tokens, args.auto_scale, args.dry), rows), 1):
            preds.append(p)
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}  {time.time() - t0:.0f}s")
    m = metrics(preds)
    m["wall_seconds"] = round(time.time() - t0, 1)
    out = EVAL_DIR / f"{args.name}.json"
    out.write_text(json.dumps({"metrics": m, "args": {**vars(args), "model": model}, "predictions": preds},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m["by_source"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
