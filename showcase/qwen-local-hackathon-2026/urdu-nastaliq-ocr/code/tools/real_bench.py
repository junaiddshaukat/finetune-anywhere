"""Read every image in data/my_photos/ (with a .txt next to it) with every model of a RUNNING demo app
(`make demo`), exactly as the browser would (auto-enlarge included), and print CER/WER per model.

    python tools/real_bench.py [--url http://127.0.0.1:8001] [--out outputs/logs/real_bench.md]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr.config import MY_PHOTOS_DIR  # noqa: E402


def post(url: str, fields: dict | None = None, file: bytes | None = None, js: dict | None = None):
    if js is not None:
        data, ct = json.dumps(js).encode(), "application/json"
    else:
        b, data = uuid.uuid4().hex, b""
        for k, v in (fields or {}).items():
            data += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        if file:
            data += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\n'
                     f'Content-Type: image/png\r\n\r\n').encode() + file + b"\r\n"
        data += f"--{b}--\r\n".encode()
        ct = f"multipart/form-data; boundary={b}"
    with urllib.request.urlopen(urllib.request.Request(url, data, {"Content-Type": ct}), timeout=1800) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--out", default="outputs/logs/real_bench.md")
    args = ap.parse_args()
    models = json.load(urllib.request.urlopen(args.url + "/api/models"))["models"]
    items = []
    for txt in sorted(MY_PHOTOS_DIR.glob("*.txt")):
        img = next((txt.with_suffix(e) for e in (".png", ".jpg", ".jpeg") if txt.with_suffix(e).exists()), None)
        if img:
            items.append((txt.stem, img, txt.read_text(encoding="utf-8")))
    lines = [f"Real images in data/my_photos ({len(items)}), CER / WER per model (lower is better)\n",
             "| image | " + " | ".join(f"{m['name']} ({m['tag']})" for m in models) + " |",
             "|---|" + "---|" * len(models)]
    tot = {m["key"]: [0, 0, 0, 0] for m in models}
    for name, img, ref in items:
        up = post(args.url + "/api/upload", file=img.read_bytes())
        cells = []
        for m in models:
            try:
                r = post(args.url + "/api/read", {"id": up["id"], "model": m["key"], "mode": "whole"})
                s = post(args.url + "/api/score", js={"pred": r["text"], "ref": ref})
                cells.append(f"{100 * s['cer']:.1f}% / {100 * s['wer']:.1f}%")
                t = tot[m["key"]]
                t[0] += s["char_edits"]; t[1] += s["ref_chars"]; t[2] += s["word_edits"]; t[3] += s["ref_words"]
            except Exception as e:  # keep going: one failing model must not hide the others
                cells.append(f"error: {str(e)[:40]}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| **all** | " + " | ".join(
        f"**{100 * tot[m['key']][0] / max(1, tot[m['key']][1]):.1f}% / {100 * tot[m['key']][2] / max(1, tot[m['key']][3]):.1f}%**"
        for m in models) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
