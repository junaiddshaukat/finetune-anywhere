"""Build one manifest of (line image, Urdu text) pairs from all downloaded sources.

Output:
  data/processed/images/<source>/<id>.png
  data/processed/manifest.jsonl  rows: {id, image, text, source, kind, split, license}

Splits:
  nastaliq_synth : each sentence is rendered 5x with different damage, so we split by
                   SENTENCE (never the same text in train and test). Their val shard -> our test.
  qaari_news     : official train/validation/test.
  book_1908      : test only (evaluation-only licence; never trained on).
  handwriting    : official writer-disjoint splits.
  my_photos      : test only - your own photos in data/my_photos/ (see README there).

Usage:
    python -m uocr.data.build
    python -m uocr.data.build --max-synth-train 30000      # smaller/faster
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from PIL import Image  # noqa: E402

from uocr.config import IMAGES_DIR, MANIFEST_PATH, MY_PHOTOS_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402
from uocr.data.download import LICENSES  # noqa: E402
from uocr.text import normalize, urdu_ratio  # noqa: E402

TEXT_KEYS = ("text", "transcription", "ground_truth", "gt", "label", "sentence", "caption", "truth")
IMAGE_KEYS = ("image", "img", "line_image", "png")


def _text_key(row: dict) -> str | None:
    for k in TEXT_KEYS:
        if k in row and isinstance(row[k], str):
            return k
    for k, v in row.items():  # fallback: first string column with Urdu in it
        if isinstance(v, str) and urdu_ratio(v) > 0.5:
            return k
    return None


def _image_bytes(row: dict) -> bytes | None:
    for k in IMAGE_KEYS:
        v = row.get(k)
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
    for v in row.values():
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
    return None


def _save_png(data_or_path, dst: Path) -> bool:
    if dst.exists():
        return True
    try:
        img = Image.open(io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path)
        img = img.convert("L") if img.mode not in ("L", "RGB") else img
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, optimize=True)
        return True
    except Exception as e:
        print(f"  skip unreadable image {dst.name}: {e}")
        return False


MANUAL_DIR = RAW_DIR / "manual"


def _cached(repo: str, filename: str) -> str | None:
    """A file from the HF cache, or one you downloaded by hand into data/raw/manual/<repo name>/<path>."""
    manual = MANUAL_DIR / repo.split("/")[-1] / filename
    if manual.exists():
        return str(manual)
    p = try_to_load_from_cache(repo, filename, repo_type="dataset")
    return p if isinstance(p, str) else None


def _iter_parquet(path: str, batch_size: int = 2000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _row(source: str, uid: str, text: str, split: str, kind: str) -> dict:
    return {"id": f"{source}_{uid}", "image": f"images/{source}/{uid}.png", "text": text,
            "source": source, "kind": kind, "split": split, "license": LICENSES[source]}


# ---------------------------------------------------------------- sources
def build_synth_rows() -> list[dict]:
    """Synthetic lines fetched with `download --slow` (rows API) instead of parquet shards."""
    root = RAW_DIR / "synth_rows"
    rows, train_texts, per_split = [], set(), Counter()
    for src_split in ("train", "test"):
        index = root / f"{src_split}.jsonl"
        if not index.exists():
            continue
        for line in index.open(encoding="utf-8"):
            m = json.loads(line)
            text = normalize(m["text"])
            src = root / src_split / m["filename"]
            if not text or not src.exists():
                continue
            if src_split == "train":
                bucket = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
                split = "val" if bucket < 3 else "train"   # sentence-level split
            else:
                if text in train_texts:
                    continue                                # no sentence overlap with train
                split = "test"
            uid = hashlib.md5(m["filename"].encode()).hexdigest()[:14]
            if _save_png(src, IMAGES_DIR / "nastaliq_synth" / f"{uid}.png"):
                rows.append(_row("nastaliq_synth", uid, text, split, "printed_synthetic"))
                per_split[split] += 1
                if split == "train":
                    train_texts.add(text)
    if rows:
        print(f"[nastaliq_synth/rows] {dict(per_split)}")
    return rows


def build_synth(max_train: int, max_val: int, max_test: int) -> list[dict]:
    if (RAW_DIR / "synth_rows").exists():
        return build_synth_rows()
    repo = "PuristanLabs1/urdu-ocr-1M"
    train_files = [f"nastaliq/train-{i:05d}-of-00010.parquet" for i in range(10)]
    train_paths = [p for f in train_files if (p := _cached(repo, f))]
    test_path = _cached(repo, "nastaliq/val-00000-of-00001.parquet")
    if not train_paths:
        print("[nastaliq_synth] not downloaded - skipping")
        return []

    rows, seen_train_text, per_split = [], set(), Counter()
    for path in train_paths:
        for r in _iter_parquet(path):
            text = normalize(r[_text_key(r)] or "")
            if not text:
                continue
            # sentence-level split: all 5 renderings of a sentence go to the same split
            bucket = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
            split = "val" if bucket < 2 else "train"
            cap = max_val if split == "val" else max_train
            if per_split[split] >= cap:
                continue
            uid = hashlib.md5((r.get("filename") or text + str(per_split[split])).encode()).hexdigest()[:14]
            if _save_png(_image_bytes(r), IMAGES_DIR / "nastaliq_synth" / f"{uid}.png"):
                rows.append(_row("nastaliq_synth", uid, text, split, "printed_synthetic"))
                per_split[split] += 1
                if split == "train":
                    seen_train_text.add(text)
        if per_split["train"] >= max_train and per_split["val"] >= max_val:
            break

    if test_path:
        for r in _iter_parquet(test_path):
            if per_split["test"] >= max_test:
                break
            text = normalize(r[_text_key(r)] or "")
            if not text or text in seen_train_text:  # no sentence overlap with train
                continue
            uid = "t" + hashlib.md5((r.get("filename") or text).encode()).hexdigest()[:13]
            if _save_png(_image_bytes(r), IMAGES_DIR / "nastaliq_synth" / f"{uid}.png"):
                rows.append(_row("nastaliq_synth", uid, text, "test", "printed_synthetic"))
                per_split["test"] += 1
    print(f"[nastaliq_synth] {dict(per_split)}")
    return rows


def _fetch_url(url: str, dest: Path, attempts: int = 4) -> bytes | None:
    """Qaari stores image URLs, not bytes. Download once, cache on disk."""
    import time
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return data
        except Exception:  # OSError, IncompleteRead, timeouts... all mean "retry"
            time.sleep(3 * (i + 1))
    return None


def build_qaari(max_per_split: dict[str, int]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    repo = "oddadmix/qaari-0.1-ocr-urdu-news-dataset-small"
    cache = RAW_DIR / "qaari_images"
    rows = []
    for hf_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        path = _cached(repo, f"data/{hf_split}-00000-of-00001.parquet")
        if not path:
            continue
        items = []
        for i, r in enumerate(_iter_parquet(path, 500)):
            if len(items) >= max_per_split[split]:
                break
            key = _text_key(r)
            text = normalize(r[key]) if key else ""
            if text:
                items.append((f"{split}{i:05d}", text, r))

        def load(item):
            uid, text, r = item
            img = r.get("image")
            if isinstance(img, str) and img.startswith("http"):
                return item, _fetch_url(img, cache / Path(img).name)
            return item, _image_bytes(r)

        with ThreadPoolExecutor(16) as pool:
            for n, ((uid, text, _), data) in enumerate(pool.map(load, items), 1):
                if data and _save_png(data, IMAGES_DIR / "qaari_news" / f"{uid}.png"):
                    rows.append(_row("qaari_news", uid, text, split, "printed_real"))
                if n % 200 == 0:
                    print(f"  [qaari_news] {split}: {n}/{len(items)} images")
    print(f"[qaari_news] {Counter(r['split'] for r in rows)}" if rows else "[qaari_news] not downloaded - skipping")
    return rows


def build_book_1908() -> list[dict]:
    meta = RAW_DIR / "book_1908" / "lines" / "metadata.jsonl"
    if not meta.exists():
        print("[book_1908] not downloaded - skipping")
        return []
    # metadata.jsonl has one record per TEXT line, but each image is a scan band that holds
    # several lines ({file_name, band, line_in_band, text}). Group them: one item per image,
    # its lines joined in reading order.
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for line in meta.open(encoding="utf-8"):
        m = json.loads(line)
        if "file_name" not in m or "text" not in m:
            print(f"  [book_1908] unexpected metadata keys: {list(m)}")
            return []
        by_file[m["file_name"]].append((int(m.get("line_in_band") or 0), m["text"]))
    rows = []
    for fname, lines in by_file.items():
        src = RAW_DIR / "book_1908" / "lines" / Path(fname).name
        uid = Path(fname).stem
        text = normalize("\n".join(t for _, t in sorted(lines)))
        if text and src.exists() and _save_png(src, IMAGES_DIR / "book_1908" / f"{uid}.png"):
            rows.append(_row("book_1908", uid, text, "test", "printed_real"))
    print(f"[book_1908] {len(rows)} test images, {sum(map(len, by_file.values()))} text lines (evaluation only)")
    return rows


def build_handwriting() -> list[dict]:
    root = RAW_DIR / "handwriting"
    if not (root / "train.csv").exists():
        print("[handwriting] not downloaded - skipping (use download --handwriting)")
        return []
    # CSV columns: id, gender, age, csv, img, line, img_<feature>..., writer_id
    #   img  = relative path of the rendered line image, e.g. Dataset/Data_1000/img/img_0000_0.png
    #   line = Urdu transcription. The img_<feature> columns are pen-trajectory feature maps
    #   with the SAME file names in other folders - never use those as the line image.
    # The archive may extract with or without an extra top-level folder, so find the base dir.
    bases = [root] + [p for p in root.iterdir() if p.is_dir()]
    rows, missing = [], 0
    for split in ("train", "val", "test"):
        with (root / f"{split}.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if "img" not in cols or "line" not in cols:
                print(f"  [handwriting] unexpected columns {cols[:8]}...; edit build_handwriting()")
                return []
            for m in reader:
                rel = (m.get("img") or "").replace("\\", "/")
                text = normalize(m.get("line") or "")
                path = next((b / rel for b in bases if rel and (b / rel).exists()), None)
                if path is None or not text:
                    missing += 1
                    continue
                uid = f"{split}_{path.parent.parent.name}_{path.stem}"  # e.g. train_Data_1000_img_0000_0
                if _save_png(path, IMAGES_DIR / "handwriting" / f"{uid}.png"):
                    rows.append(_row("handwriting", uid, text, split, "handwritten"))
    if missing:
        print(f"  [handwriting] {missing} CSV rows without an image file or text (skipped)")
    print(f"[handwriting] {Counter(r['split'] for r in rows)}")
    return rows


def build_my_photos() -> list[dict]:
    """Your own test photos: data/my_photos/<name>.jpg + <name>.txt (the true Urdu text)."""
    rows = []
    for img in sorted(MY_PHOTOS_DIR.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        uid = img.stem
        if _save_png(img, IMAGES_DIR / "my_photos" / f"{uid}.png"):
            r = _row_any("my_photos", uid, normalize(txt.read_text(encoding="utf-8")), "test", "printed_real_photo")
            rows.append(r)
    if rows:
        print(f"[my_photos] {len(rows)} test photos")
    return rows


def _row_any(source, uid, text, split, kind):
    return {"id": f"{source}_{uid}", "image": f"images/{source}/{uid}.png", "text": text,
            "source": source, "kind": kind, "split": split, "license": "own data"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-synth-train", type=int, default=60000)
    ap.add_argument("--max-synth-val", type=int, default=1000)
    ap.add_argument("--max-synth-test", type=int, default=2000)
    ap.add_argument("--max-qaari-train", type=int, default=20000,
                    help="news images are fetched one by one from URLs; lower this on slow internet")
    ap.add_argument("--max-qaari-test", type=int, default=1000)
    args = ap.parse_args()

    qaari_caps = {"train": args.max_qaari_train, "val": max(1, args.max_qaari_test // 2), "test": args.max_qaari_test}
    rows = (build_synth(args.max_synth_train, args.max_synth_val, args.max_synth_test)
            + build_qaari(qaari_caps) + build_book_1908() + build_handwriting() + build_my_photos())
    if not rows:
        raise SystemExit("No data. Run: python -m uocr.data.download")

    # drop exact duplicate (image, text) pairs and empty texts
    seen, unique = set(), []
    for r in rows:
        if r["text"] and (r["image"], r["text"]) not in seen:
            seen.add((r["image"], r["text"]))
            unique.append(r)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    table = defaultdict(Counter)
    for r in unique:
        table[r["source"]][r["split"]] += 1
    print(f"\n{'source':16s} {'kind':20s} {'train':>7s} {'val':>6s} {'test':>6s}")
    kinds = {r["source"]: r["kind"] for r in unique}
    for s, c in table.items():
        print(f"{s:16s} {kinds[s]:20s} {c['train']:7d} {c['val']:6d} {c['test']:6d}")
    print(f"\nmanifest ({len(unique)} lines) -> {MANIFEST_PATH}")
    print("See the data: python -m uocr.data.preview   (then open data/preview/index.html)")


if __name__ == "__main__":
    main()
