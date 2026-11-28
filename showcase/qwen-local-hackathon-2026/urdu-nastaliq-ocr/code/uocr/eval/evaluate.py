"""Benchmark Urdu OCR: base Qwen vs fine-tuned Qwen vs Claude, per test source.

Metrics (lower is better): CER = character error rate, WER = word error rate.
Also: exact-line match rate, and share of output that is Urdu script.

Usage:
    python -m uocr.eval.evaluate --name base --per-source 200
    python -m uocr.eval.evaluate --name finetuned --adapter outputs/qwen3vl2b-urdu-ocr-lora --per-source 200
    python -m uocr.eval.evaluate --name claude --backend claude --per-source 100     # costs API credits
    python -m uocr.eval.evaluate --compare base finetuned claude
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from uocr.config import BASE_MODEL, DEFAULT_MAX_SIDE, MANIFEST_PATH, OUTPUTS_DIR, PROCESSED_DIR
from uocr.text import corpus_cer, corpus_wer, normalize, strict, urdu_ratio

EVAL_DIR = OUTPUTS_DIR / "eval"


def load_test(per_source: int | None, sources: list[str] | None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in MANIFEST_PATH.open(encoding="utf-8")]
    by_src = defaultdict(list)
    for r in rows:
        if r["split"] == "test" and (not sources or r["source"] in sources):
            by_src[r["source"]].append(r)
    out, rng = [], random.Random(seed)
    for src, g in sorted(by_src.items()):
        rng.shuffle(g)
        out += g[:per_source] if per_source else g
    return out


# Sources whose crops overlap their neighbours: the image shows (parts of) lines that belong to
# another crop's truth. book_1908 = 8 overlapping bands per column. For these, cer/wer use
# semi-global alignment (text read before/after the reference is free); cer_plain keeps plain CER.
OVERLAPPING_CROP_SOURCES = {"book_1908"}


def metrics(preds: list[dict]) -> dict:
    def block(ps):
        p, r = [x["pred"] for x in ps], [x["text"] for x in ps]
        ifx = [x["source"] in OVERLAPPING_CROP_SOURCES for x in ps]
        if any(ifx) and not all(ifx):  # mixed (overall): sum per-item edits with the right rule each
            def cc(norm, word=False):
                f = corpus_wer if word else corpus_cer
                tot = lambda s: sum(len(norm(x).split()) if word else len(norm(x)) for x in s)  # noqa: E731
                e = sum(f([a], [b], norm, infix=i) * max(1, tot([b])) for a, b, i in zip(p, r, ifx))
                return e / max(1, tot(r))
        else:
            def cc(norm, word=False):
                return (corpus_wer if word else corpus_cer)(p, r, norm, infix=all(ifx))
        return {"n": len(ps), "cer": cc(normalize), "wer": cc(normalize, True),
                "cer_strict": cc(strict), "wer_strict": cc(strict, True),
                "cer_plain": corpus_cer(p, r),
                "missing_rate": sum(not x.strip() for x in p) / max(1, len(ps)),
                "exact_match": sum(normalize(a) == normalize(b) for a, b in zip(p, r)) / max(1, len(ps)),
                "urdu_script_ratio": sum(urdu_ratio(a) for a in p) / max(1, len(ps)),
                "sec_per_line": sum(x["seconds"] for x in ps) / max(1, len(ps))}
    by_src = defaultdict(list)
    for x in preds:
        by_src[x["source"]].append(x)
    return {"overall": block(preds), "by_source": {s: block(v) for s, v in sorted(by_src.items())}}


def run_qwen(rows, adapter, base_model, max_side, batch_size, max_new_tokens=256, instruction=None) -> list[dict]:
    from uocr.inference import UrduReader
    from uocr.prompts import INSTRUCTION

    reader = UrduReader(base_model, adapter, max_side=max_side, instruction=instruction or INSTRUCTION)
    preds = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        t0 = time.time()
        outs = reader.read_lines([PROCESSED_DIR / r["image"] for r in chunk], batch_size=batch_size,
                                 max_new_tokens=max_new_tokens)
        dt = (time.time() - t0) / len(chunk)
        preds += [{**r, "pred": o, "seconds": dt} for r, o in zip(chunk, outs)]
        print(f"  {len(preds)}/{len(rows)}")
    return preds


QAARI_BASE = "Qwen/Qwen2-VL-2B-Instruct"
QAARI_ADAPTER = "oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct"


def qaari_adapter_dir() -> str:
    """Qaari's adapter was saved from an Unsloth 4-bit Qwen2-VL with the OLD transformers module
    names (`model.layers.N`, `visual.blocks.N`). transformers 5 nests them as
    `model.language_model.layers.N` and `model.visual.blocks.N`, so PEFT would silently load
    nothing. Convert once: rename the keys and point the config at the bf16 base."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    out = OUTPUTS_DIR / "qaari_adapter_tf5"
    if (out / "adapter_model.safetensors").exists():
        return str(out)
    src = Path(snapshot_download(QAARI_ADAPTER, allow_patterns=["adapter_*"]))
    sd = load_file(src / "adapter_model.safetensors")
    renamed = {}
    for k, v in sd.items():
        k2 = k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
        k2 = k2.replace("base_model.model.visual.", "base_model.model.model.visual.")
        renamed[k2] = v
    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, out / "adapter_model.safetensors")
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = QAARI_BASE
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Converted Qaari adapter ({len(renamed)} tensors) -> {out}")
    return str(out)


def run_qaari(rows, max_side, batch_size, max_new_tokens=256) -> list[dict]:
    """Existing Urdu OCR fine-tune (Qwen2-VL-2B + LoRA). The main competitor to beat.
    Uses Qaari's own documented prompt (QAARI_INSTRUCTION) for a fair baseline.
    NOTE: Qaari was likely trained on the qaari_news data - its score there is not a fair comparison."""
    from uocr.prompts import QAARI_INSTRUCTION

    return run_qwen(rows, qaari_adapter_dir(), QAARI_BASE, max_side, batch_size, max_new_tokens,
                    instruction=QAARI_INSTRUCTION)


def _find_tesseract() -> str:
    import shutil
    for cand in (shutil.which("tesseract"), r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        if cand and __import__("os").path.exists(cand):
            return cand
    raise SystemExit("Tesseract not found. Windows: winget install UB-Mannheim.TesseractOCR "
                     "(then add urd.traineddata from github.com/tesseract-ocr/tessdata_best to its tessdata folder). "
                     "Linux: sudo apt install tesseract-ocr tesseract-ocr-urd")


MULTILINE_SOURCES = {"qaari_news", "book_1908", "my_photos"}


def run_tesseract(rows) -> list[dict]:
    """Classic OCR control. psm 7 = single text line; psm 6 = block of text (pages, multi-line crops)."""
    import subprocess

    exe = _find_tesseract()
    preds = []
    for r in rows:
        psm = 6 if r["source"] in MULTILINE_SOURCES else 7
        t0 = time.time()
        res = subprocess.run([exe, str(PROCESSED_DIR / r["image"]), "stdout", "-l", "urd", "--psm", str(psm)],
                             capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0 and not preds:
            raise SystemExit(f"tesseract failed: {res.stderr.strip()[:300]}")
        preds.append({**r, "pred": res.stdout.strip(), "seconds": time.time() - t0})
    return preds


def run_claude(rows, max_side) -> list[dict]:
    import base64
    import io

    import anthropic

    from uocr.modeling import load_image
    from uocr.prompts import INSTRUCTION, clean_output

    client = anthropic.Anthropic(max_retries=4)
    preds = []
    for r in rows:
        img = load_image(PROCESSED_DIR / r["image"], max_side)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        t0, out = time.time(), ""
        try:
            resp = client.beta.messages.create(
                model="claude-opus-5-5", max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.standard_b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": INSTRUCTION}]}],
            )
            if resp.stop_reason != "refusal":
                out = clean_output(next((b.text for b in resp.content if b.type == "text"), ""))
        except anthropic.APIStatusError as e:
            print(f"  API error {e.status_code} on {r['id']}")
        except anthropic.APIConnectionError:
            print(f"  network error on {r['id']}")
        preds.append({**r, "pred": out, "seconds": time.time() - t0})
        if len(preds) % 20 == 0:
            print(f"  {len(preds)}/{len(rows)}")
    return preds


def print_compare(names: list[str]) -> None:
    res = {n: json.loads((EVAL_DIR / f"{n}.json").read_text(encoding="utf-8"))["metrics"] for n in names}
    sources = sorted({s for m in res.values() for s in m["by_source"]})

    def cell(m, key):
        return "-" if m is None else f"{m[key] * 100:.1f}%" if key != "sec_per_line" else f"{m[key]:.2f}s"

    print("CER (lower is better)\n| test set | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| **overall** | " + " | ".join(cell(res[n]["overall"], "cer") for n in names) + " |")
    for s in sources:
        print(f"| {s} | " + " | ".join(cell(res[n]["by_source"].get(s), "cer") for n in names) + " |")
    print("\n| metric (overall) | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for k in ("cer_strict", "cer_plain", "wer", "wer_strict", "exact_match", "missing_rate", "urdu_script_ratio", "sec_per_line"):
        print(f"| {k} | " + " | ".join(cell(res[n]["overall"], k) for n in names) + " |")


def main() -> None:
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # live progress when redirected to a log file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--backend", choices=["qwen", "qaari", "tesseract", "claude"], default="qwen")
    ap.add_argument("--adapter")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--per-source", type=int, default=200, help="test lines per source (0 = all)")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="raise for full pages (news pages need ~1500)")
    ap.add_argument("--prompt", choices=["default", "qaari"], default="default",
                    help="instruction for qwen backend (qaari = Qaari's own prompt, e.g. for a Qaari fine-tune)")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()

    if args.compare:
        print_compare(args.compare)
        return
    if not args.name:
        ap.error("--name is required")
    rows = load_test(args.per_source or None, args.sources)
    print(f"Evaluating '{args.name}' on {len(rows)} test lines")
    if args.backend == "claude":
        preds = run_claude(rows, args.max_side)
    elif args.backend == "qaari":
        preds = run_qaari(rows, args.max_side, args.batch_size, args.max_new_tokens)
    elif args.backend == "tesseract":
        preds = run_tesseract(rows)
    else:
        from uocr.prompts import PROMPTS
        preds = run_qwen(rows, args.adapter, args.base_model, args.max_side, args.batch_size, args.max_new_tokens,
                         instruction=PROMPTS[args.prompt])
    m = metrics(preds)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{args.name}.json"
    out.write_text(json.dumps({"metrics": m, "args": vars(args), "predictions": preds},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2))
    print(f"-> {out}\nSee predictions side by side: python -m uocr.eval.report --name {args.name}")


if __name__ == "__main__":
    main()
