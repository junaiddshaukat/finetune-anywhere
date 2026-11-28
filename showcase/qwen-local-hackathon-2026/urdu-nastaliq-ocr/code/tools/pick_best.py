"""Pick the best of several evals on the test items they all contain.

    python tools/pick_best.py v2_q8_0_glyph48 v3_q8_0_glyph48

Score = mean over sources of the pooled CER (so a big source can't drown the small real-photo ones).
book_1908 is reported but not used for the pick: it is dominated by repetition loops, not reading skill.
Prints a table, then the winning eval name alone on the last line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uocr.eval.evaluate import EVAL_DIR, metrics  # noqa: E402

SKIP = {"book_1908"}


def main(names: list[str]) -> None:
    preds = {n: {p["id"]: p for p in json.loads((EVAL_DIR / f"{n}.json").read_text(encoding="utf-8"))["predictions"]}
             for n in names}
    common = set.intersection(*(set(v) for v in preds.values()))
    table, score = {}, {}
    for n in names:
        m = metrics([preds[n][i] for i in sorted(common)])["by_source"]
        table[n] = {s: v["cer"] for s, v in m.items()}
        score[n] = sum(v for s, v in table[n].items() if s not in SKIP) / max(1, sum(s not in SKIP for s in table[n]))
    srcs = sorted(next(iter(table.values())))
    print(f"{len(common)} common test items")
    print("| eval | " + " | ".join(srcs) + " | pick score |\n|---|" + "---|" * (len(srcs) + 1))
    for n in names:
        print(f"| {n} | " + " | ".join(f"{100 * table[n][s]:.1f}%" for s in srcs) + f" | {100 * score[n]:.2f}% |")
    print(min(names, key=lambda n: score[n]))


if __name__ == "__main__":
    main(sys.argv[1:])
