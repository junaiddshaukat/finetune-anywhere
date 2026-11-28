"""Download Urdu OCR datasets from Hugging Face into data/raw/ (with retries for flaky networks).

Sources (license):
  nastaliq_synth  PuristanLabs1/urdu-ocr-1M, Nastaliq part (MIT). ~50k lines per shard.
  qaari_news      oddadmix/qaari-0.1-ocr-urdu-news-dataset-small (MIT). Urdu news OCR.
  book_1908       vettedlance/urdu-nastaliq-1908-eval-sample (EVALUATION-ONLY licence:
                  never train on it). Real 1908 printed book lines, human-transcribed.
  handwriting     saad2002/ouhd-l-online-urdu-nastaliq-handwriting (CC BY 4.0). 2,403
                  handwritten lines, 311 writers. 1.9 GB archive - opt-in with --handwriting.

Usage:
    python -m uocr.data.download                       # synth (1 shard) + news + 1908 book
    python -m uocr.data.download --synth-shards 3      # more training data (~150k lines)
    python -m uocr.data.download --handwriting         # also the handwriting set
"""
from __future__ import annotations

import argparse
import os
import tarfile
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP; more robust on unstable links

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from uocr.config import HF_TOKEN, RAW_DIR  # noqa: E402

LICENSES = {
    "nastaliq_synth": "MIT - PuristanLabs1/urdu-ocr-1M (https://huggingface.co/datasets/PuristanLabs1/urdu-ocr-1M)",
    "qaari_news": "MIT - oddadmix/qaari-0.1-ocr-urdu-news-dataset-small",
    "book_1908": "Vettedlance Evaluation-Only Licence v1.0 - EVALUATION ONLY, do not train or redistribute "
                 "(https://huggingface.co/datasets/vettedlance/urdu-nastaliq-1908-eval-sample)",
    "handwriting": "CC BY 4.0 - OUHD-L, DOI 10.5281/zenodo.20642162",
}


def _retry(fn, *args, attempts: int = 8, **kwargs):
    for i in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # network errors surface as many different exception types
            if i == attempts:
                raise
            wait = min(60, 5 * i)
            print(f"  attempt {i}/{attempts} failed ({type(e).__name__}: {str(e)[:80]}); retry in {wait}s")
            time.sleep(wait)


def _get(repo: str, filename: str) -> str:
    return _retry(hf_hub_download, repo, filename, repo_type="dataset", token=HF_TOKEN)


def download_synth(shards: int) -> None:
    repo = "PuristanLabs1/urdu-ocr-1M"
    files = [f"nastaliq/train-{i:05d}-of-00010.parquet" for i in range(shards)] + \
            ["nastaliq/val-00000-of-00001.parquet"]
    for f in files:
        print(f"  {f}")
        _get(repo, f)
    print(f"[nastaliq_synth] {len(files)} files ready (in the HF cache)")


ROWS_API = "https://datasets-server.huggingface.co/rows"


def download_synth_rows(n_train: int, n_test: int) -> None:
    """Fetch only N lines through the HF rows API (no 100 MB parquet chunks). For slow internet.

    Writes data/raw/synth_rows/<split>/<filename>.png + data/raw/synth_rows/<split>.jsonl
    """
    import json
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def get(url: str) -> bytes:
        def once():
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        return _retry(once)

    for split, hf_split, n in (("test", "val", n_test), ("train", "train", n_train)):
        out_dir = RAW_DIR / "synth_rows" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = RAW_DIR / "synth_rows" / f"{split}.jsonl"
        done = {json.loads(l)["filename"] for l in index_path.open(encoding="utf-8")} if index_path.exists() else set()
        offset = len(done)  # resume where we stopped
        with index_path.open("a", encoding="utf-8") as idx, ThreadPoolExecutor(8) as pool:
            while offset < n:
                q = urllib.parse.urlencode({"dataset": "PuristanLabs1/urdu-ocr-1M", "config": "nastaliq",
                                            "split": hf_split, "offset": offset, "length": min(100, n - offset)})
                rows = [r["row"] for r in json.loads(get(f"{ROWS_API}?{q}"))["rows"]]
                if not rows:
                    break

                def fetch(row):
                    dest = out_dir / row["filename"]
                    if not dest.exists():
                        dest.write_bytes(get(row["image"]["src"]))
                    return row

                for row in pool.map(fetch, rows):
                    if row["filename"] not in done:
                        idx.write(json.dumps({"filename": row["filename"], "text": row["text"]},
                                             ensure_ascii=False) + "\n")
                        done.add(row["filename"])
                idx.flush()
                offset += len(rows)
                print(f"  [synth {split}] {offset}/{n}", flush=True)
    print("[nastaliq_synth via rows API] ready")


def download_qaari() -> None:
    """Only the small parquet files (text + image URLs). Images are fetched by `build`."""
    import urllib.request

    repo = "oddadmix/qaari-0.1-ocr-urdu-news-dataset-small"
    out = RAW_DIR / "manual" / repo.split("/")[-1] / "data"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("test", "validation", "train"):
        dest = out / f"{split}-00000-of-00001.parquet"
        if dest.exists() and _is_parquet(dest):
            continue
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/{split}-00000-of-00001.parquet"

        def once():
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
        _retry(once)
        print(f"  {dest.name}")
    print("[qaari_news] ready")


def _is_parquet(path) -> bool:
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def download_book_1908() -> None:
    path = _retry(snapshot_download, "vettedlance/urdu-nastaliq-1908-eval-sample", repo_type="dataset",
                  token=HF_TOKEN, local_dir=RAW_DIR / "book_1908")
    print(f"[book_1908] ready -> {path}  (evaluation only)")


def download_handwriting() -> None:
    repo = "saad2002/ouhd-l-online-urdu-nastaliq-handwriting"
    out = RAW_DIR / "handwriting"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "val.csv", "test.csv", "DATA_CARD.md", "LICENSE"):
        _get_to(repo, f, out)
    marker = out / ".extracted"
    if not marker.exists():
        tar_path = _get(repo, "ouhdl_v1.0_core.tar.gz")
        print("  extracting archive (1.9 GB)...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(out, filter="data")
        marker.touch()
    print(f"[handwriting] ready -> {out}")


def _get_to(repo: str, filename: str, out_dir) -> None:
    import shutil
    shutil.copy(_get(repo, filename), out_dir / filename)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-shards", type=int, default=1, help="Nastaliq train shards (1-10, ~50k lines each)")
    ap.add_argument("--handwriting", action="store_true")
    ap.add_argument("--slow", action="store_true",
                    help="slow internet: fetch only --synth-train/--synth-test lines via the rows API")
    ap.add_argument("--synth-train", type=int, default=15000)
    ap.add_argument("--synth-test", type=int, default=1000)
    ap.add_argument("--only", nargs="*", choices=["nastaliq_synth", "qaari_news", "book_1908", "handwriting"])
    args = ap.parse_args()

    todo = args.only or ["nastaliq_synth", "qaari_news", "book_1908"] + (["handwriting"] if args.handwriting else [])
    failed = []
    for name in todo:
        print(f"== {name} ==", flush=True)
        try:  # one failing source must not stop the others
            {"nastaliq_synth": (lambda: download_synth_rows(args.synth_train, args.synth_test)) if args.slow
                               else (lambda: download_synth(args.synth_shards)),
             "qaari_news": download_qaari,
             "book_1908": download_book_1908,
             "handwriting": download_handwriting}[name]()
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:150]} - re-run the same command to resume")
            failed.append(name)
    if failed:
        print(f"\nSources to retry: {failed}")

    (RAW_DIR / "ATTRIBUTION.txt").write_text(
        "Datasets used:\n" + "\n".join(f"- {k}: {v}" for k, v in LICENSES.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
