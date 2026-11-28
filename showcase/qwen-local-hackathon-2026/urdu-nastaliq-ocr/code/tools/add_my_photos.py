"""Add the images in data/my_photos/ (png/jpg + same-name .txt with the true text) to the processed
manifest as test-only source `my_photos`, without rebuilding the other sources.

    python tools/add_my_photos.py

Never used for training (train_sources in the configs does not list my_photos).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr.config import MY_PHOTOS_DIR, PROCESSED_DIR  # noqa: E402

MANIFEST = PROCESSED_DIR / "manifest.jsonl"


def main() -> None:
    out_dir = PROCESSED_DIR / "images" / "my_photos"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for txt in sorted(MY_PHOTOS_DIR.glob("*.txt")):
        img = next((txt.with_suffix(e) for e in (".png", ".jpg", ".jpeg") if txt.with_suffix(e).exists()), None)
        if img is None:
            continue
        text = "\n".join(l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip())
        Image.open(img).convert("RGB").save(out_dir / f"{txt.stem}.png")
        rows.append({"id": f"my_photos_{txt.stem}", "image": f"images/my_photos/{txt.stem}.png", "text": text,
                     "source": "my_photos", "kind": "real_photo", "split": "test", "license": "own data"})
    kept = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines() if '"source": "my_photos"' not in l]
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text("\n".join(kept + [json.dumps(r, ensure_ascii=False) for r in rows]) + "\n", encoding="utf-8")
    os.replace(tmp, MANIFEST)
    print(f"my_photos: {len(rows)} test images in the manifest")


if __name__ == "__main__":
    main()
