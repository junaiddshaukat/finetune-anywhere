"""Split a photo/scan of a printed Urdu page into text-line images.

The model is trained on single lines, so full pages are cut into lines first, read line
by line, then joined. This uses a simple horizontal-projection method that works well on
clean, straight, single-column print (books, notices, newspaper columns). Tilted photos,
multi-column layouts and mixed images need a real layout model (out of scope for the MVP).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def segment_lines(img: Image.Image, min_line_height: int = 12, pad_ratio: float = 0.25,
                  work_width: int = 1600) -> list[Image.Image]:
    """Return line crops top-to-bottom. If no lines are found, returns [img]."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = min(1.0, work_width / img.width)
    small = img.resize((int(img.width * scale), int(img.height * scale))) if scale < 1 else img
    gray = np.asarray(small.convert("L"))
    ink = gray <= _otsu_threshold(gray)         # True where dark text (Otsu class 0 is <= t)
    if ink.mean() > 0.5:                         # white text on dark background
        ink = ~ink
    profile = ink.sum(axis=1)
    thresh = max(1, 0.02 * profile.max())
    rows = profile > thresh

    # group consecutive text rows into bands
    bands, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(rows)))

    # merge bands separated by tiny gaps (Nastaliq dots/marks above and below the line)
    merged = []
    heights = [b - a for a, b in bands] or [0]
    gap_merge = max(2, int(np.median(heights) * 0.35))
    for a, b in bands:
        if merged and a - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if (b - a) >= min_line_height * scale]
    if not merged:
        return [img]

    crops = []
    for a, b in merged:
        pad = int((b - a) * pad_ratio)
        y0, y1 = max(0, a - pad), min(len(rows), b + pad)
        cols = np.where(ink[y0:y1].any(axis=0))[0]
        x0, x1 = (cols[0], cols[-1] + 1) if len(cols) else (0, small.width)
        xpad = int((b - a) * 0.5)
        box = (max(0, x0 - xpad) / scale, y0 / scale, min(small.width, x1 + xpad) / scale, y1 / scale)
        crops.append(img.crop(tuple(int(v) for v in box)))
    return crops


def glyph_height(img: Image.Image, pct: float = 80) -> float | None:
    """Text size: the 80th-percentile height (px) of ink blobs (ligatures). Dots and diacritics
    fall below the percentile; ruled notebook lines and borders are removed first so the letters
    touching them are not swallowed. None if there is too little text to measure."""
    import cv2

    g = np.asarray(img.convert("L"))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # ink = 255
    if bw.mean() > 127:                                                         # light text on dark
        bw = 255 - bw
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 6), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 2)))
    bw = cv2.subtract(bw, cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [ch for _, _, cw, ch, area in stats[1:] if area >= 6 and ch >= 3 and cw <= 0.6 * w and ch <= 0.95 * h]
    return float(np.percentile(hs, pct)) if len(hs) >= 3 else None


def scale_for_reading(img: Image.Image, target_glyph: float = 48, max_side: int = 2048,
                      max_scale: float = 4.0) -> Image.Image:
    """Enlarge images whose letters are too small for the vision encoder (32x32 px patches).

    Measured on real photos/scans (not the synthetic test set, which reads fine at 13 px):
    a 2-line printed Nastaliq couplet went 14% -> 5% CER and a notebook handwriting photo
    54% -> 8% when letters were enlarged to ~50 px. Never shrinks.
    """
    gh = glyph_height(img)
    if not gh or gh >= target_glyph:
        return img
    s = min(max_scale, target_glyph / gh, max_side / max(img.size))
    if s <= 1.05:
        return img
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def prepare_for_reading(path_or_img, max_side: int = 1024, auto_scale: bool = True) -> Image.Image:
    """The image as the model should see it: the training resize (load_image), then, for densely
    packed multi-line images, an enlargement taken from the ORIGINAL pixels (not the shrunk copy)."""
    from uocr.modeling import load_image

    base = load_image(path_or_img, max_side)
    if not auto_scale:
        return base
    scaled = scale_for_reading(base)
    if scaled is base:
        return base
    src = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return load_image(src, 4096).resize(scaled.size, Image.LANCZOS)
