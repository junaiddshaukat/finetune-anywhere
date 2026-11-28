"""Render our existing sentences as clean text-line images in Noto Nastaliq Urdu (SIL OFL), a second
real Nastaliq typeface next to the Jameel Noori of nastaliq_synth. Uses headless Edge/Chrome as the
text shaper (PIL here has no libraqm, so it cannot join Urdu letters). Labels are exact by construction.

    python tools/render_noto_lines.py --train 16000 --val 400 --test 800
    python tools/render_noto_lines.py --preview 12            # only a montage in outputs/preview/

New source `nastaliq_noto` in data/processed/manifest.jsonl (sentences keep their original split, so a
test sentence is never a training sentence). Needs `pip install websocket-client`.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uocr.config import PROCESSED_DIR  # noqa: E402

SOURCE = "nastaliq_noto"
LICENSE = "MIT texts (PuristanLabs1/urdu-ocr-1M) rendered in Noto Nastaliq Urdu (SIL OFL 1.1)"
MANIFEST = PROCESSED_DIR / "manifest.jsonl"
FONT = (ROOT / "app" / "static" / "fonts" / "NotoNastaliqUrdu.ttf").as_uri()
EDGES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
         r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
         r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9335
BATCH = 40


class Browser:
    def __init__(self):
        exe = next(p for p in EDGES if os.path.exists(p))
        self.prof = ROOT / "outputs" / "tmp_edge_profile"
        self.proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={PORT}",
                                      f"--user-data-dir={self.prof}", "--window-size=1800,1400",
                                      "--hide-scrollbars", "--allow-file-access-from-files", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        self.n = 0
        self.cdp("Page.enable")

    def cdp(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr):
        r = self.cdp("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()


def render_batch(b: Browser, items: list[dict]) -> list[Image.Image]:
    """items: {text, size, weight}. Returns one tight grayscale image per item."""
    rows = "".join(
        f'<div class="l" style="font-size:{it["size"]}px;font-weight:{it["weight"]}">{it["text"]}</div>'
        for it in items)
    html = ("<!doctype html><meta charset=utf-8><style>@font-face{font-family:N;src:url('%s')}"
            "body{margin:0;background:#fff}.l{font-family:N;direction:rtl;white-space:nowrap;width:max-content;"
            "padding:14px 22px;margin:6px 10px;background:#fff;color:#000;line-height:2.2}</style>%s" % (FONT, rows))
    path = ROOT / "outputs" / "tmp_render.html"
    path.write_text(html, encoding="utf-8")
    b.cdp("Page.navigate", url=path.as_uri())
    for _ in range(100):
        if b.js("document.readyState") == "complete":
            break
        time.sleep(0.1)
    b.js("document.fonts.ready.then(() => true)")
    rects = json.loads(b.js("""JSON.stringify([...document.querySelectorAll('.l')].map(e => {
        const r = e.getBoundingClientRect();
        return [r.left + scrollX, r.top + scrollY, r.width, r.height]; }))"""))
    out = []
    for x, y, w, h in rects:
        shot = b.cdp("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                     clip={"x": x, "y": y, "width": w, "height": h, "scale": 1})["data"]
        out.append(Image.open(io.BytesIO(base64.b64decode(shot))).convert("L"))
    return out


def sentences(split: str, n: int, rng: random.Random) -> list[str]:
    seen, out = set(), []
    for l in MANIFEST.open(encoding="utf-8"):
        if '"source": "nastaliq_synth"' not in l:
            continue
        r = json.loads(l)
        t = " ".join(r["text"].split())
        if r["split"] == split and 6 <= len(t) <= 90 and t not in seen and "<" not in t and "&" not in t:
            seen.add(t)
            out.append(t)
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=16000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    plan = {"train": args.train, "val": args.val, "test": args.test}
    if args.preview:
        plan = {"train": args.preview}

    img_dir = PROCESSED_DIR / "images" / SOURCE
    img_dir.mkdir(parents=True, exist_ok=True)
    b, new = Browser(), []
    try:
        for split, n in plan.items():
            texts = sentences(split, n, rng)
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i + BATCH]
                items = [{"text": t, "size": rng.randint(30, 64), "weight": rng.choice([400, 400, 500, 600])}
                         for t in chunk]
                for k, (t, im) in enumerate(zip(chunk, render_batch(b, items))):
                    uid = f"{split}_{i + k:06d}"
                    if args.preview:
                        im.save(ROOT / "outputs" / "preview" / f"noto_{i + k}.png")
                        continue
                    im.save(img_dir / f"{uid}.png", optimize=True)
                    new.append({"id": f"{SOURCE}_{uid}", "image": f"images/{SOURCE}/{uid}.png", "text": t,
                                "source": SOURCE, "kind": "printed_synthetic", "split": split, "license": LICENSE})
                print(f"[{SOURCE}] {split} {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    finally:
        b.close()
    if args.preview:
        return
    kept = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines() if f'"source": "{SOURCE}"' not in l]
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text("\n".join(kept + [json.dumps(r, ensure_ascii=False) for r in new]) + "\n", encoding="utf-8")
    os.replace(tmp, MANIFEST)   # atomic: readers never see a half-written manifest
    print(f"manifest: {len(kept)} other rows + {len(new)} {SOURCE} rows")


if __name__ == "__main__":
    main()
