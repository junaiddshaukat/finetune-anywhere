"""The single instruction used for training, eval and the app."""
from __future__ import annotations

INSTRUCTION = (
    "Read the Urdu text in this image and write it out exactly, in Urdu script. "
    "Keep the original words and punctuation. Output only the text, nothing else."
)


# Qaari 0.1's documented prompt (model card usage example). Used only for the Qaari baseline,
# so it is evaluated with the instruction it was trained/shipped with.
QAARI_INSTRUCTION = (
    "Below is the image of one page of a document, as well as some raw textual content that was "
    "previously extracted for it. Just return the plain text representation of this document as if "
    "you were reading it naturally. Do not hallucinate."
)


PROMPTS = {"default": INSTRUCTION, "qaari": QAARI_INSTRUCTION}


def build_messages(image=None, target: str | None = None, instruction: str = INSTRUCTION) -> list[dict]:
    """Qwen-VL chat messages. `image` may be a PIL image or a path."""
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})
    msgs = [{"role": "user", "content": content}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def clean_output(text: str) -> str:
    """Strip code fences / quotes a base model may wrap around the text, and collapse generation loops."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1] if "\n" in text else text.strip("`")
    return trim_repetition(text.strip().strip('"').strip("«»").strip())


def trim_repetition(text: str, max_period: int = 12, min_repeats: int = 4, min_char_repeats: int = 30) -> str:
    """Collapse generation loops to one copy, keeping any text around them.

    - the same 1-12 words repeated >= 4 times in a row ("عزتی عزتی عزتی ...")
    - the same 1-6 characters repeated >= 30 times inside a word ("2000000000...")
    Real Urdu repeats a word at most twice ("ساتھ ساتھ") and real numbers stay far below 30
    repeated digits, so normal text is left alone. Nothing after a loop is dropped.
    """
    import re

    text = re.sub(r"(.{1,6}?)\1{%d,}" % (min_char_repeats - 1), r"\1", text)
    words = text.split(" ")
    out, i, n = [], 0, len(words)
    while i < n:
        for p in range(1, min(max_period, (n - i) // min_repeats) + 1):
            block = words[i:i + p]
            r = 1
            while words[i + r * p:i + (r + 1) * p] == block:
                r += 1
            if r >= min_repeats:
                out += block
                i += r * p
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)
