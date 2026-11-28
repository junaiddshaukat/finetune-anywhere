"""Urdu OCR web app: read your own images with several models side by side, check them against the
true text, keep them as test data.

Run (see the Makefile):
    make demo         # our fine-tune (GGUF Q8_0) vs Tesseract vs Qaari + our data, side by side
    make app          # our fine-tune alone, in PyTorch on the GPU (the exact eval model)
    make app-local    # our fine-tune alone, through an already running llama-server (`make llama`)

Environment (.env or Makefile):
    UOCR_MODELS        comma-separated keys from MODELS below (default: ours_torch)
    UOCR_ADAPTER_PATH  LoRA adapter of our fine-tune, for ours_torch
    UOCR_LLAMA_URL     llama-server to use for ours_gguf. If nothing answers there and UOCR_LLAMA_BIN
                       is set, the app starts its own llama-server on that port and stops it on exit.
    UOCR_LLAMA_BIN     folder with llama-server(.exe); UOCR_LLAMA_NGL layers on the GPU (99 = all)
    UOCR_GGUF          GGUF file for the started llama-server (default outputs/gguf/urdu-ocr-2b-Q8_0.gguf)
    UOCR_MOCK=1        fake models, for UI work without a GPU
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from uocr.config import BASE_MODEL, DEFAULT_MAX_SIDE, MY_PHOTOS_DIR, OUTPUTS_DIR, ROOT
from uocr.pages import prepare_for_reading, segment_lines
from uocr.prompts import INSTRUCTION, QAARI_INSTRUCTION, clean_output
from uocr.text import levenshtein, normalize

STATIC = Path(__file__).parent / "static"
MOCK = os.getenv("UOCR_MOCK", "0") == "1"
ADAPTER = os.getenv("UOCR_ADAPTER_PATH", "outputs/qwen3vl2b-urdu-ocr-lora")
LLAMA_URL = os.getenv("UOCR_LLAMA_URL", "http://127.0.0.1:8090").rstrip("/")
LLAMA_BIN = os.getenv("UOCR_LLAMA_BIN", "")
LLAMA_NGL = os.getenv("UOCR_LLAMA_NGL", "0")
GGUF = os.getenv("UOCR_GGUF", "outputs/gguf/urdu-ocr-2b-v3-Q8_0.gguf")          # latest fine-tune
GGUF_OLD = os.getenv("UOCR_GGUF_OLD", "outputs/gguf/urdu-ocr-2b-Q8_0.gguf")      # first fine-tune (v1)
_host, _port = LLAMA_URL.rsplit(":", 1)
LLAMA_URL_OLD = f"{_host}:{int(_port) + 1}"                                       # its own llama-server
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Enlarge densely packed multi-line images before reading (uocr.pages.scale_for_reading). UOCR_AUTO_SCALE=0 to disable.
AUTO_SCALE = os.getenv("UOCR_AUTO_SCALE", "1") == "1"
LINE_TOKENS, PAGE_TOKENS = 256, 2048   # a whole page can be 1-2k tokens; 256 cut pages off

app = FastAPI(title="Urdu OCR")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")   # already sized by prepare_for_reading
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- readers
# Each reader: .detail (shown on hover), .read(img, max_new_tokens, single_line) -> str.

class MockReader:
    detail = "UOCR_MOCK=1, no model loaded"

    def __init__(self, delay: float = 0.6):
        self.delay = delay

    def read(self, img, max_new_tokens, single_line):
        time.sleep(self.delay)
        return "(نمونہ) یہ ایک نمونہ سطر ہے"


class TorchReader:
    """A Qwen-VL model + unmerged LoRA in bf16 on the GPU, exactly as evaluated."""

    def __init__(self, base: str, adapter: str | None, instruction: str = INSTRUCTION):
        from uocr.inference import UrduReader

        base_path = str(_abs(base)) if _abs(base).exists() else base   # local folder or HF id
        adapter_path = str(_abs(adapter)) if adapter else None
        if adapter_path and not Path(adapter_path).exists():
            raise RuntimeError(f"adapter not found: {adapter}")
        # max_side 4096: images arrive already sized by prepare_for_reading; don't shrink them again
        self.reader = UrduReader(base_path, adapter_path, instruction=instruction, max_side=4096)
        self.detail = f"{Path(base).name} + {Path(adapter).name if adapter else 'no adapter'} · bf16 · PyTorch"

    def read(self, img, max_new_tokens, single_line):
        return self.reader.read_lines([img], max_new_tokens=max_new_tokens, batch_size=1)[0]


def _llama_props(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/props", timeout=3) as r:
            return json.load(r)
    except OSError:
        return None


def _start_llama(url: str, gguf_path: str) -> None:
    """Start llama-server on the port in `url` and wait until the model is loaded."""
    exe = Path(LLAMA_BIN) / ("llama-server.exe" if os.name == "nt" else "llama-server")
    gguf = _abs(gguf_path)
    # the vision projector belongs to the same build: urdu-ocr-2b[-v2]-Q8_0.gguf -> urdu-ocr-2b[-v2]-mmproj-f16.gguf
    mmproj = gguf.parent / (re.sub(r"-(Q\d\w*|F16|BF16)$", "", gguf.stem, flags=re.I) + "-mmproj-f16.gguf")
    for f in (exe, gguf, mmproj):
        if not f.exists():
            raise RuntimeError(f"not found: {f}")
    port = url.rsplit(":", 1)[-1]
    log = open(OUTPUTS_DIR / "logs" / f"llama_app_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(exe), "-m", str(gguf), "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", port,
         "-c", "8192", "-ngl", LLAMA_NGL, "-t", "8", "--jinja"], stdout=log, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited, see outputs/logs/llama_app_{port}.log")
        if _llama_props(url):
            return
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready in 2 min")


class LlamaCppReader:
    """The quantized GGUF behind llama-server (OpenAI-compatible chat API)."""

    def __init__(self, url: str = LLAMA_URL, gguf: str = GGUF):
        self.url = url
        props = _llama_props(url)
        if props is None:
            if not LLAMA_BIN:
                raise RuntimeError(f"llama-server is not reachable at {url} (start it with `make llama`)")
            _start_llama(url, gguf)
            props = _llama_props(url) or {}
        model = Path(props.get("model_path", "model.gguf")).name
        where = "GPU" if LLAMA_BIN and LLAMA_NGL != "0" else "CPU" if LLAMA_BIN else "external server"
        self.detail = f"{model} · llama.cpp · {where} · {url}"

    def read(self, img, max_new_tokens, single_line):
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(img)}},
                {"type": "text", "text": INSTRUCTION}]}],
            "temperature": 0, "max_tokens": max_new_tokens, "repeat_penalty": 1.05,
        }
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return clean_output(json.load(r)["choices"][0]["message"]["content"])


class TesseractReader:
    """Tesseract 5 with urd.traineddata, the classic OCR control. Same page modes as the eval."""

    def __init__(self):
        from uocr.eval.evaluate import _find_tesseract

        try:
            self.exe = _find_tesseract()
        except SystemExit as e:  # the eval helper exits; here we want an error in the UI
            raise RuntimeError(str(e)) from None
        ver = subprocess.run([self.exe, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
        self.detail = f"{ver.strip()} · urd · CPU"

    def read(self, img, max_new_tokens, single_line):
        # psm 7 = one text line, psm 6 = block of text. Judge by shape: a "line" crop from the
        # segmenter can still hold several lines, and psm 7 then returns nothing.
        psm = "7" if img.width > (3 if single_line else 6) * img.height else "6"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "in.png"
            img.save(path)
            res = subprocess.run([self.exe, str(path), "stdout", "-l", "urd", "--psm", psm],
                                 capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[:200] or "tesseract failed")
        return res.stdout.strip()


# ---------------------------------------------------------------- model registry
# `eval` names the outputs/eval/<name>.json whose overall CER is shown next to the model.

def _qaari_original():
    """Qaari 0.1 exactly as released (its adapter on Qwen2-VL-2B), with its own prompt. Not trained on our data."""
    from uocr.eval.evaluate import QAARI_BASE, qaari_adapter_dir

    return TorchReader(QAARI_BASE, str(qaari_adapter_dir()), QAARI_INSTRUCTION)


MODELS = {
    "ours_gguf": dict(name="Our fine-tune (v3)", tag="Qwen3-VL-2B · GGUF Q8_0 · latest", eval="v3_q8_0_glyph48",
                      make=lambda: LlamaCppReader(LLAMA_URL, GGUF)),
    "ours_v1_gguf": dict(name="Our first fine-tune (v1)", tag="Qwen3-VL-2B · GGUF Q8_0", eval="gguf_q8_0_glyph48",
                         make=lambda: LlamaCppReader(LLAMA_URL_OLD, GGUF_OLD)),
    "qaari": dict(name="Qaari 0.1 (original)", tag="Qwen2-VL-2B · PyTorch", eval="qaari", make=_qaari_original),
    "ours_torch": dict(name="Our fine-tune", tag="Qwen3-VL-2B · PyTorch", eval="finetuned",
                       make=lambda: TorchReader(BASE_MODEL, ADAPTER)),
    "tesseract": dict(name="Tesseract", tag="Tesseract 5 · urd", eval="tesseract",
                      make=TesseractReader),
    "qaari_ft": dict(name="Qaari + our data", tag="Qwen2-VL-2B · PyTorch", eval="qaari_ft",
                     make=lambda: TorchReader("outputs/qaari-qwen2-vl-merged", "outputs/qaari-ft-lora",
                                              QAARI_INSTRUCTION)),
}
ENABLED = [k.strip() for k in os.getenv("UOCR_MODELS", "ours_torch").split(",") if k.strip()]
for k in ENABLED:
    if k not in MODELS:
        raise SystemExit(f"UOCR_MODELS: unknown model '{k}' (choose from {', '.join(MODELS)})")


_REF_IDS: set | None = None


def _test_cer(name: str) -> float | None:
    """Pooled CER on the test items that ALL four demo models were evaluated on (about 360), with the same
    loop collapse the app applies. Different eval runs sampled different items, so a chip measured on each
    model's own sample would not be comparable."""
    global _REF_IDS
    try:
        from uocr.eval.evaluate import metrics
        from uocr.prompts import trim_repetition

        def load(n):
            return json.loads((OUTPUTS_DIR / "eval" / f"{n}.json").read_text(encoding="utf-8"))["predictions"]

        if _REF_IDS is None:
            _REF_IDS = set.intersection(*({p["id"] for p in load(n)} for n in
                                          ("v3_q8_0_glyph48", "gguf_q8_0_glyph48", "qaari", "qaari_ft")))
        rows = [{**p, "pred": trim_repetition(p["pred"])} for p in load(name) if p["id"] in _REF_IDS]
        return metrics(rows)["overall"]["cer"] if len(rows) == len(_REF_IDS) else None
    except Exception:
        return None


class Slot:
    """One model: loaded once (in the background at startup), used by one request at a time."""

    def __init__(self, key: str):
        spec = MODELS[key]
        self.key, self.name, self.tag = key, spec["name"], spec["tag"]
        self.test_cer = _test_cer(spec["eval"])
        self._make = (lambda: MockReader()) if MOCK else spec["make"]
        self.reader, self.error, self.state = None, None, "loading"
        self._init_lock, self.run_lock = threading.Lock(), threading.Lock()

    def get(self):
        with self._init_lock:
            if self.reader is None and self.error is None:
                try:
                    self.reader, self.state = self._make(), "ready"
                except Exception as e:  # report it in the UI instead of crashing the server
                    self.error, self.state = str(e), "error"
        if self.error:
            raise HTTPException(503, f"{self.name}: {self.error}")
        return self.reader

    def info(self) -> dict:
        detail = self.error or (self.reader.detail if self.reader else "Loading…")
        return {"key": self.key, "name": self.name, "tag": self.tag, "state": self.state,
                "detail": detail, "test_cer": self.test_cer}


SLOTS = {k: Slot(k) for k in ENABLED}


def _warm(slot: Slot) -> None:
    try:
        slot.get()
    except HTTPException:
        pass


@app.on_event("startup")
def _load_models_in_background():
    # torch loads are not thread-safe enough to overlap reliably; load one after another
    threading.Thread(target=lambda: [_warm(s) for s in SLOTS.values()], daemon=True).start()


# ---------------------------------------------------------------- uploads, kept so every model can read the same image

_uploads: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_crops: dict[str, list[Image.Image]] = {}
_uploads_lock = threading.Lock()
MAX_KEPT = 64


def _keep(data: bytes, filename: str) -> str:
    uid = hashlib.sha1(data).hexdigest()[:12]
    with _uploads_lock:
        _uploads[uid] = (data, filename)
        _uploads.move_to_end(uid)
        while len(_uploads) > MAX_KEPT:
            old, _ = _uploads.popitem(last=False)
            _crops.pop(old, None)
    return uid


def _open(data: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image")


def _lines(uid: str, img: Image.Image) -> list[Image.Image]:
    """Line crops are cut once per image, so every model reads exactly the same crops."""
    with _uploads_lock:
        if uid not in _crops:
            _crops[uid] = segment_lines(img)
        return _crops[uid]


# ---------------------------------------------------------------- API

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models():
    return {"mock": MOCK, "models": [s.info() for s in SLOTS.values()]}


@app.post("/api/upload")
async def upload(image: UploadFile = File(...)):
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    img = _open(data)
    return {"id": _keep(data, image.filename or "image"), "width": img.width, "height": img.height}


_pool = ThreadPoolExecutor(8)


@app.post("/api/read")
async def read(id: str = Form(...), model: str = Form(...), mode: str = Form("whole")):
    if model not in SLOTS:
        raise HTTPException(404, f"Unknown model '{model}'")
    if id not in _uploads:
        raise HTTPException(404, "Image is no longer in memory; add it again")
    import asyncio

    def run():
        slot = SLOTS[model]
        reader = slot.get()
        img = _open(_uploads[id][0])
        with slot.run_lock:  # one image per model at a time; different models run in parallel
            t0 = time.time()
            if mode == "lines":
                crops = _lines(id, img)
                texts = [reader.read(prepare_for_reading(c, DEFAULT_MAX_SIDE, False), LINE_TOKENS, True) for c in crops]
            else:  # whole image: how the model saw news pages and lines in training
                crops = [img]
                side = 4096 if isinstance(reader, TesseractReader) else DEFAULT_MAX_SIDE  # no token budget there
                texts = [reader.read(prepare_for_reading(img, side, AUTO_SCALE), PAGE_TOKENS, False)]
            return {"model": model, "mode": mode, "lines": len(crops), "text": "\n".join(texts),
                    "seconds": round(time.time() - t0, 2)}

    try:
        return await asyncio.get_running_loop().run_in_executor(_pool, run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{SLOTS[model].name} failed: {e}")


class ScoreIn(BaseModel):
    pred: str
    ref: str


@app.post("/api/score")
def score(body: ScoreIn):
    """CER/WER with the same normalisation as the eval, plus a word-level diff for display."""
    from rapidfuzz.distance import Levenshtein

    pred, ref = normalize(body.pred), normalize(body.ref)
    if not ref:
        raise HTTPException(400, "Enter the true text first")
    pw, rw = pred.split(), ref.split()
    diff = []
    for op in Levenshtein.opcodes(rw, pw):   # word diff: char spans would break Nastaliq joining
        diff.append({"op": op.tag, "ref": " ".join(rw[op.src_start:op.src_end]),
                     "pred": " ".join(pw[op.dest_start:op.dest_end])})
    char_edits, word_edits = levenshtein(pred, ref), levenshtein(pw, rw)
    return {"cer": char_edits / max(1, len(ref)), "wer": word_edits / max(1, len(rw)),
            "char_edits": char_edits, "ref_chars": len(ref), "word_edits": word_edits, "ref_words": len(rw),
            "diff": diff}


@app.post("/api/save")
def save(id: str = Form(...), ref: str = Form(...), name: str = Form("")):
    """Keep an image + its true text as a my_photos TEST item (never used for training)."""
    if id not in _uploads:
        raise HTTPException(404, "Image is no longer in memory; add it again")
    if not normalize(ref):
        raise HTTPException(400, "Enter the true text first")
    data, filename = _uploads[id]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name or Path(filename).stem).strip("_")[:40] or "photo"
    stem = f"{stem}_{id[:6]}"
    _open(data).save(MY_PHOTOS_DIR / f"{stem}.png")
    (MY_PHOTOS_DIR / f"{stem}.txt").write_text(ref.strip() + "\n", encoding="utf-8")
    n = len(list(MY_PHOTOS_DIR.glob("*.txt")))
    return {"ok": True, "path": f"data/my_photos/{stem}.png", "count": n}
