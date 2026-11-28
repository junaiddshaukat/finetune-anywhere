"""Turn the tablet handwriting (OUHD-L: clean pen strokes on white) into notebook-photo-like images.

Real handwriting reaches the model as phone photos of paper: ruled lines, ink colour, uneven light,
shadows, blur, tilt, JPEG. OUHD-L has none of that, and on a real notebook photo the fine-tune
dropped to 19-54% CER. This builds a new source, `handwriting_paper`, from the SAME writer-disjoint
splits (train images from train writers only; test images from test writers only), with 1-3 lines
of one writer stacked on one ruled page. Labels join the lines with "\\n", top to bottom.

    python tools/make_notebook_photos.py --variants 3          # ~3 images per train line group
    python tools/make_notebook_photos.py --preview 12          # write a few to outputs/preview/

Rows are (re)written into data/processed/manifest.jsonl under source `handwriting_paper`.
OUHD-L is CC BY 4.0: derived images keep that licence and attribution.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr.config import PROCESSED_DIR  # noqa: E402

# --from handwriting: OUHD-L tablet lines -> notebook photos.  --from synth: synthetic printed
# Nastaliq lines (train split only for train) -> book pages / photos of print, 1-5 lines each.
STYLES = {
    "handwriting": dict(source="handwriting_paper", kind="handwritten_photo_aug", ruled=0.75, line_h=(38, 90),
                        sizes=([1, 2, 3], [0.40, 0.35, 0.25]),
                        license="CC BY 4.0 - OUHD-L (doi:10.5281/zenodo.20642162), augmented into notebook photos"),
    "noto": dict(source="print_paper_noto", kind="printed_photo_aug", ruled=0.08, line_h=(26, 80),
                 from_source="nastaliq_noto", filter_ink=False,
                 sizes=([1, 2, 3, 4, 5], [0.15, 0.25, 0.25, 0.2, 0.15]),
                 license="MIT texts rendered in Noto Nastaliq Urdu (SIL OFL 1.1), stacked into pages / photos"),
    "synth": dict(source="print_paper", from_source="nastaliq_synth", filter_ink=True, kind="printed_photo_aug", ruled=0.08, line_h=(26, 80),
                  sizes=([1, 2, 3, 4, 5], [0.15, 0.25, 0.25, 0.2, 0.15]),
                  license="MIT - PuristanLabs1/urdu-ocr-1M, stacked into pages / photos"),
}
MANIFEST = PROCESSED_DIR / "manifest.jsonl"

PAPERS = [(246, 245, 240), (240, 233, 214), (228, 226, 220), (214, 212, 206), (236, 238, 242), (226, 212, 172)]
INKS = [(20, 20, 25), (25, 35, 95), (15, 30, 70), (55, 55, 60), (30, 30, 45)]
RULES = [(120, 150, 205), (110, 110, 115), (95, 125, 190), (140, 140, 150)]   # notebook rules show up dark in photos


def ink_mask(path: Path, line_h: int, rng: random.Random) -> np.ndarray:
    """Alpha mask (0..1) of the strokes, cropped to the ink and scaled to `line_h` px tall."""
    g = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    ys, xs = np.where(g < 200)
    g = g[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    a = np.clip((235 - g.astype(np.float32)) / 235, 0, 1)   # paper grain of noisy sources stays faint
    s = line_h / a.shape[0]
    a = cv2.resize(a, (max(8, int(a.shape[1] * s)), line_h), interpolation=cv2.INTER_AREA)
    pen = rng.choice([0, 0, 1, 1, 2])                       # thicker pens than the tablet stylus
    if pen:
        a = cv2.dilate(a, np.ones((pen + 1, pen + 1), np.uint8))
    return np.clip(a * rng.uniform(0.85, 1.0), 0, 1)


def solid_ink(path: Path, min_ratio: float = 0.6) -> bool:
    """False for eroded / broken lines (the source's "thin" damage variant): there, most ink pixels
    are pale. Their labels still list every letter, which teaches the model to guess unseen words."""
    g = np.asarray(Image.open(path).convert("L"))
    return ((g < 110).sum() + 1) / ((g < 185).sum() + 1) >= min_ratio


def page(paths: list[Path], rng: random.Random, style: dict) -> Image.Image:
    line_h = rng.randint(*style["line_h"])
    masks = [ink_mask(p, line_h, rng) for p in paths]
    pitch = int(line_h * rng.uniform(1.25, 1.7))
    margin_x, margin_y = rng.randint(10, 60), rng.randint(int(pitch * 0.3), pitch)
    slack = rng.randint(0, 80)                              # free space for per-line indents
    W = max(m.shape[1] for m in masks) + 2 * margin_x + slack
    H = margin_y * 2 + pitch * len(masks)

    # paper: base colour, fine grain, slow uneven lighting / shadow
    paper = np.array(rng.choice(PAPERS), np.float32) + rng.uniform(-8, 8)
    canvas = np.ones((H, W, 3), np.float32) * paper
    canvas += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1.5, 6), (H, W, 1))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    grad = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    light = 1 - rng.uniform(0.0, 0.35) * (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    if rng.random() < 0.3:   # soft shadow blob (hand / phone)
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.3, 0.8) * max(W, H)
        light *= 1 - rng.uniform(0.1, 0.3) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    canvas *= light[..., None]

    # ruled lines under the text baselines (+ sometimes a margin line)
    if rng.random() < style["ruled"]:
        rule = np.array(rng.choice(RULES), np.float32)
        th = rng.choice([1, 1, 2])
        for k in range(-1, len(masks) + 1):
            y = margin_y + k * pitch + int(line_h * rng.uniform(0.85, 1.05))
            if 0 <= y < H - th:
                canvas[y:y + th] = canvas[y:y + th] * 0.35 + rule * 0.65
        if rng.random() < 0.3:
            x = W - rng.randint(15, max(16, margin_x))
            canvas[:, x:x + th] = canvas[:, x:x + th] * 0.4 + np.array((215, 120, 120), np.float32) * 0.6

    # ink, right-aligned like Urdu writing, each line a little off
    ink = np.array(rng.choice(INKS if style["ruled"] > 0.5 else [INKS[0], INKS[3], INKS[4]]), np.float32)
    for k, m in enumerate(masks):
        y0 = margin_y + k * pitch + rng.randint(-line_h // 10, line_h // 10)
        x1 = W - margin_x - rng.randint(0, slack)               # never pushes the line off the page
        x0 = x1 - m.shape[1]
        y0 = max(0, min(H - m.shape[0], y0))
        reg = canvas[y0:y0 + m.shape[0], x0:x1]
        canvas[y0:y0 + m.shape[0], x0:x1] = reg * (1 - m[..., None]) + ink * m[..., None]

    img = np.clip(canvas, 0, 255).astype(np.uint8)
    # camera: tilt / perspective, blur, resolution loss, JPEG, sometimes grayscale
    a = rng.uniform(-2.5, 2.5)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        d = 0.03 * min(W, H)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), borderMode=cv2.BORDER_REPLICATE)
    # keep it legible: a page a person could still read (scaled to the text size)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.6 + line_h / 120))
    if rng.random() < 0.4:
        s = rng.uniform(0.65, 0.9)
        small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    out = Image.fromarray(img)
    if rng.random() < 0.25:
        out = out.convert("L").convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(45, 92))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def groups(rows: list[dict], rng: random.Random, style: dict) -> list[list[dict]]:
    """Consecutive lines of the same writer (or random synthetic lines) per page."""
    by_writer = defaultdict(list)
    for r in rows:
        by_writer[r["writer"]].append(r)
    out = []
    for lines in by_writer.values():
        i = 0
        while i < len(lines):
            n = rng.choices(*style["sizes"])[0]
            out.append(lines[i:i + n])
            i += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=int, default=3, help="pages per line group (train/val); test gets 1")
    ap.add_argument("--preview", type=int, default=0, help="only write N examples to outputs/preview/")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from", dest="src", choices=list(STYLES), default="handwriting")
    ap.add_argument("--max-lines", type=int, default=12000, help="synth: lines used for train (val/test: 1/40, 1/20)")
    args = ap.parse_args()

    style = STYLES[args.src]
    SOURCE, LICENSE = style["source"], style["license"]
    rng = random.Random(args.seed)
    split_rows = {}
    if args.src == "handwriting":
        import csv
        root = Path(__file__).resolve().parents[1] / "data" / "raw" / "handwriting"
        for split in ("train", "val", "test"):
            with open(root / f"{split}.csv", encoding="utf-8") as f:
                split_rows[split] = [{"img": root / r["img"], "text": r["line"].strip(), "writer": r["writer_id"]}
                                     for r in csv.DictReader(f) if r["line"].strip() and (root / r["img"]).exists()]
    else:   # synthetic lines keep their manifest split, so test pages only use test sentences
        rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if f'"{style["from_source"]}"' in l]
        caps = {"train": args.max_lines, "val": args.max_lines // 40, "test": args.max_lines // 20}
        for split, cap in caps.items():
            sel = [r for r in rows if r["split"] == split]
            rng.shuffle(sel)
            if style["filter_ink"]:
                sel = [r for r in sel[:int(cap * 1.8)] if solid_ink(PROCESSED_DIR / r["image"])]
            sel = sel[:cap]
            split_rows[split] = [{"img": PROCESSED_DIR / r["image"], "text": r["text"].strip(), "writer": 0}
                                 for r in sel]

    if args.preview:
        out_dir = Path(__file__).resolve().parents[1] / "outputs" / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, g in enumerate(groups(split_rows["train"], rng, style)[:args.preview]):
            page([r["img"] for r in g], rng, style).save(out_dir / f"{SOURCE}_{k}.jpg")
        print(f"wrote {args.preview} previews to {out_dir}")
        return

    img_dir = PROCESSED_DIR / "images" / SOURCE
    img_dir.mkdir(parents=True, exist_ok=True)
    new = []
    for split, rows in split_rows.items():
        variants = 1 if split == "test" else args.variants
        for gi, g in enumerate(groups(rows, rng, style)):
            text = "\n".join(r["text"] for r in g)
            for v in range(variants):
                uid = f"{split}_{gi:05d}_{v}"
                page([r["img"] for r in g], rng, style).save(img_dir / f"{uid}.png", optimize=True)
                new.append({"id": f"{SOURCE}_{uid}", "image": f"images/{SOURCE}/{uid}.png", "text": text,
                            "source": SOURCE, "kind": style["kind"], "split": split, "license": LICENSE})
        print(f"[{SOURCE}] {split}: {sum(r['split'] == split for r in new)} images")

    kept = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines() if f'"source": "{SOURCE}"' not in l]
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text("\n".join(kept + [json.dumps(r, ensure_ascii=False) for r in new]) + "\n", encoding="utf-8")
    os.replace(tmp, MANIFEST)   # atomic: readers never see a half-written manifest
    print(f"manifest: {len(kept)} other rows + {len(new)} {SOURCE} rows")


if __name__ == "__main__":
    main()
