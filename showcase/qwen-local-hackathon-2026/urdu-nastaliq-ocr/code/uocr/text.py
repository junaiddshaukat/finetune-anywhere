"""Urdu text normalization and OCR metrics (CER / WER).

Normalization makes visually identical text compare as equal: Arabic vs Urdu code
points for yeh/kaf/heh, zero-width characters, tatweel, extra spaces. Apply the SAME
normalization to training targets, model outputs and references.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic-script variants that render the same in Urdu text -> canonical Urdu code point
_CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH -> FARSI/URDU YEH
    "ى": "ی",  # ALEF MAKSURA -> URDU YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH
    "ه": "ہ",  # ARABIC HEH -> HEH GOAL
    "ە": "ہ",  # AE -> HEH GOAL
    "ـ": "",        # TATWEEL
    "​": "",        # zero width space
    "‌": "",        # ZWNJ
    "‍": "",        # ZWJ
    "‎": "",        # LRM
    "‏": "",        # RLM
    "﻿": "",        # BOM
}
_TRANS = str.maketrans(_CHAR_MAP)
_SPACES = re.compile(r"\s+")


def strict(text: str) -> str:
    """Strict form: only Unicode NFC + whitespace cleanup (no letter merging)."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def normalize(text: str, strip_diacritics: bool = False) -> str:
    """Lenient form: also merges Arabic/Urdu code points that render identically."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_TRANS)
    if strip_diacritics:  # zabar/zer/pesh etc. Often absent in print; optional for eval
        text = "".join(ch for ch in text if not ("ً" <= ch <= "ْ" or ch == "ٰ"))
    return _SPACES.sub(" ", text).strip()


try:  # fast C implementation (same result); pure-Python fallback below
    from rapidfuzz.distance import Levenshtein as _RFLev
except ImportError:  # pragma: no cover
    _RFLev = None


def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings or lists of words)."""
    if _RFLev is not None:
        return _RFLev.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


INFIX_FREE_CONTEXT = 1.5  # free extra output, as a multiple of the reference length


def levenshtein_infix(pred, ref, free_context: float = INFIX_FREE_CONTEXT) -> int:
    """Edits to turn a contiguous PART of `pred` into `ref` (semi-global alignment), for
    overlapping crops (book_1908 bands): the image also shows neighbouring lines that belong to
    another crop's truth, so text read before/after the reference is free - but only up to
    `free_context` x len(ref) in total (a band shows ~2.5x its own lines). Output beyond that
    counts as insertions, so repetition loops and invented text are still penalised."""
    m, allowance = len(pred), int(free_context * len(ref))
    cost, start = [0] * (m + 1), list(range(m + 1))   # any start position in pred is free
    for i, cr in enumerate(ref, 1):
        c2, s2 = [i], [0]
        for j in range(1, m + 1):
            sub = cost[j - 1] + (cr != pred[j - 1])
            dele = cost[j] + 1
            ins = c2[j - 1] + 1
            if sub <= dele and sub <= ins:
                c2.append(sub); s2.append(start[j - 1])  # noqa: E702
            elif dele <= ins:
                c2.append(dele); s2.append(start[j])  # noqa: E702
            else:
                c2.append(ins); s2.append(s2[j - 1])  # noqa: E702
        cost, start = c2, s2
    # end at any j; output outside the aligned span [start, j) is free up to the allowance
    return min(cost[j] + max(0, m - (j - start[j]) - allowance) for j in range(m + 1))


def cer(pred: str, ref: str) -> float:
    """Character error rate: edits / reference length (can exceed 1.0 for garbage output)."""
    ref, pred = normalize(ref), normalize(pred)
    return levenshtein(pred, ref) / max(1, len(ref))


def wer(pred: str, ref: str) -> float:
    ref_w, pred_w = normalize(ref).split(), normalize(pred).split()
    return levenshtein(pred_w, ref_w) / max(1, len(ref_w))


def corpus_cer(preds: list[str], refs: list[str], norm=normalize, infix: bool = False) -> float:
    """Total edits / total reference characters (the standard way to report CER)."""
    dist = levenshtein_infix if infix else levenshtein
    edits = sum(dist(norm(p), norm(r)) for p, r in zip(preds, refs))
    return edits / max(1, sum(len(norm(r)) for r in refs))


def corpus_wer(preds: list[str], refs: list[str], norm=normalize, infix: bool = False) -> float:
    dist = levenshtein_infix if infix else levenshtein
    edits = sum(dist(norm(p).split(), norm(r).split()) for p, r in zip(preds, refs))
    return edits / max(1, sum(len(norm(r).split()) for r in refs))


def urdu_ratio(text: str) -> float:
    """Share of letters that are Arabic-script (sanity check that output is Urdu)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" for c in letters) / len(letters)
