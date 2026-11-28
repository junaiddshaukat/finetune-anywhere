"""Make an HTML page to browse the dataset yourself: images next to their Urdu text.

Usage:
    python -m uocr.data.preview                  # 40 random samples per source & split
    python -m uocr.data.preview --per-group 100
Then open data/preview/index.html in your browser (double-click it).
"""
from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from statistics import mean, median

from uocr.config import MANIFEST_PATH, PREVIEW_DIR

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Urdu OCR dataset preview</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
 h1{{margin:0 0 4px}} .muted{{color:#666}}
 table.stats{{border-collapse:collapse;margin:16px 0}} .stats td,.stats th{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
 .stats th{{background:#f0f0f0}} .stats td:first-child,.stats th:first-child{{text-align:left}}
 nav a{{margin-right:12px}}
 .group{{margin-top:32px}} .card{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:10px;margin:10px 0}}
 .card img{{max-width:100%;max-height:160px;border:1px solid #eee;background:#fff}}
 .urdu{{direction:rtl;text-align:right;font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq',serif;font-size:22px;line-height:2.2;margin-top:6px}}
 .meta{{font-size:12px;color:#888}} .warn{{color:#b26a00}}
</style></head><body>
<h1>Urdu OCR dataset preview</h1>
<div class="muted">Each card: the image the model sees, then the exact text it must output. {total} lines in the manifest.</div>
<h2>Statistics</h2>{stats}
<nav>{nav}</nav>
{groups}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST_PATH.open(encoding="utf-8")]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["split"])].append(r)

    stat_rows = ["<table class='stats'><tr><th>source / split</th><th>kind</th><th>lines</th>"
                 "<th>unique texts</th><th>avg chars</th><th>median chars</th><th>max chars</th></tr>"]
    for (src, split), g in sorted(groups.items()):
        lens = [len(r["text"]) for r in g]
        stat_rows.append(
            f"<tr><td>{src} / {split}</td><td>{g[0]['kind']}</td><td>{len(g)}</td>"
            f"<td>{len({r['text'] for r in g})}</td><td>{mean(lens):.0f}</td><td>{median(lens):.0f}</td>"
            f"<td>{max(lens)}</td></tr>")
    stat_rows.append("</table>")
    chars = Counter(ch for r in rows for ch in r["text"])
    stat_rows.append(f"<div class='muted'>{len(chars)} distinct characters. Most common: "
                     + " ".join(html.escape(c) for c, _ in chars.most_common(60)) + "</div>")
    licenses = sorted({(r["source"], r["license"]) for r in rows})
    stat_rows.append("<h3>Licenses</h3><ul>" + "".join(
        f"<li><b>{html.escape(s)}</b>: {html.escape(l)}</li>" for s, l in licenses) + "</ul>")

    rng = random.Random(args.seed)
    nav, sections = [], []
    for (src, split), g in sorted(groups.items()):
        anchor = f"{src}-{split}"
        nav.append(f"<a href='#{anchor}'>{src}/{split}</a>")
        sample = rng.sample(g, min(args.per_group, len(g)))
        cards = []
        for r in sample:
            note = " <span class='warn'>(evaluation only)</span>" if "EVALUATION" in r["license"] else ""
            cards.append(
                f"<div class='card'><img loading='lazy' src='../processed/{html.escape(r['image'])}'>"
                f"<div class='urdu'>{html.escape(r['text'])}</div>"
                f"<div class='meta'>{html.escape(r['id'])} · {len(r['text'])} chars{note}</div></div>")
        sections.append(f"<div class='group' id='{anchor}'><h2>{src} / {split} "
                        f"<span class='muted'>({len(g)} lines, showing {len(sample)})</span></h2>"
                        + "".join(cards) + "</div>")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / "index.html"
    out.write_text(PAGE.format(total=len(rows), stats="".join(stat_rows), nav=" ".join(nav),
                               groups="".join(sections)), encoding="utf-8")
    print(f"Open in your browser: {out}")


if __name__ == "__main__":
    main()
