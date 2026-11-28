"""HTML page showing image | true text | model output(s) | CER, to inspect errors yourself.

Usage:
    python -m uocr.eval.report --name base finetuned       # compare runs on the same lines
Opens: outputs/eval/report_<names>.html
"""
from __future__ import annotations

import argparse
import html
import json

from uocr.config import PROCESSED_DIR
from uocr.eval.evaluate import EVAL_DIR, OVERLAPPING_CROP_SOURCES
from uocr.text import cer, levenshtein_infix, normalize


def item_cer(pred: str, row: dict) -> float:
    if row["source"] in OVERLAPPING_CROP_SOURCES:  # same rule as evaluate.metrics
        ref = normalize(row["text"])
        return levenshtein_infix(normalize(pred), ref) / max(1, len(ref))
    return cer(pred, row["text"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    runs = {n: {p["id"]: p for p in json.loads((EVAL_DIR / f"{n}.json").read_text(encoding="utf-8"))["predictions"]}
            for n in args.name}
    ids = [i for i in runs[args.name[0]] if all(i in r for r in runs.values())][: args.limit]
    rel = lambda p: (PROCESSED_DIR / p).as_uri()  # noqa: E731
    rows = []
    for i in ids:
        first = runs[args.name[0]][i]
        outs = "".join(
            f"<div class='out'><b>{html.escape(n)}</b> <span class='cer'>CER {item_cer(runs[n][i]['pred'], first) * 100:.0f}%</span>"
            f"<div class='urdu'>{html.escape(runs[n][i]['pred'])}</div></div>" for n in args.name)
        rows.append(f"<div class='card'><div class='meta'>{html.escape(first['source'])} · {html.escape(i)}</div>"
                    f"<img src='{rel(first['image'])}'><div class='out'><b>truth</b>"
                    f"<div class='urdu'>{html.escape(first['text'])}</div></div>{outs}</div>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>OCR report</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">
<style>body{{font-family:system-ui;margin:24px;background:#fafafa}}.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0}}
img{{max-width:100%;max-height:150px}}.urdu{{direction:rtl;text-align:right;font-family:'Noto Nastaliq Urdu',serif;font-size:21px;line-height:2.1}}
.out{{border-top:1px dashed #eee;padding-top:4px}}.cer{{color:#b26a00;font-size:12px}}.meta{{color:#888;font-size:12px}}</style></head>
<body><h1>OCR report: {' vs '.join(map(html.escape, args.name))}</h1>{''.join(rows)}</body></html>"""
    out = EVAL_DIR / f"report_{'_'.join(args.name)}.html"
    out.write_text(page, encoding="utf-8")
    print(f"Open in your browser: {out}")


if __name__ == "__main__":
    main()
