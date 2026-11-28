"""OCR fine-tuning dataset + collator. Loss is computed only on the Urdu transcription."""
from __future__ import annotations

import json
import random

from torch.utils.data import Dataset

from uocr.config import MANIFEST_PATH, PROCESSED_DIR
from uocr.modeling import load_image
from uocr.prompts import build_messages


def load_manifest(split: str, sources: list[str] | None = None) -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Run python -m uocr.data.build first")
    rows = [json.loads(l) for l in MANIFEST_PATH.open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == split]
    if sources:
        rows = [r for r in rows if r["source"] in sources]
    return rows


class OCRDataset(Dataset):
    def __init__(self, split: str, max_side: int, limit: int | None = None,
                 sources: list[str] | None = None, mix: dict[str, float] | None = None, seed: int = 42):
        rows = load_manifest(split, sources)
        rng = random.Random(seed)
        if mix:  # oversample small real/handwritten sets, e.g. {"qaari_news": 3}
            out = []
            for r in rows:
                w = mix.get(r["source"], 1.0)
                out += [r] * int(w) + ([r] if rng.random() < (w - int(w)) else [])
            rows = out
        rng.shuffle(rows)
        self.rows = rows[:limit] if limit else rows
        self.max_side = max_side

    def drop_too_long(self, tokenizer, max_length: int, prompt_tokens: int = 64, px_per_token: int = 32) -> int:
        """Drop examples whose image + text tokens would exceed max_length (some news pages are
        ~3.7k text tokens). Truncating them instead would teach the model to stop mid-page."""
        from PIL import Image

        def image_tokens(path) -> int:  # one token per 32x32 px (Qwen3-VL) or 28x28 px (Qwen2-VL)
            w, h = Image.open(path).size
            s = min(1.0, self.max_side / max(w, h))
            return -(-int(w * s) // px_per_token) * -(-int(h * s) // px_per_token)

        cache, keep = {}, []
        for r in self.rows:
            if r["id"] not in cache:
                n_text = len(tokenizer(r["text"], add_special_tokens=False)["input_ids"])
                cache[r["id"]] = n_text + image_tokens(PROCESSED_DIR / r["image"]) + prompt_tokens
            if cache[r["id"]] <= max_length:
                keep.append(r)
        dropped = len(self.rows) - len(keep)
        self.rows = keep
        self.lengths = [cache[r["id"]] for r in keep]  # estimated tokens, for length-grouped batching
        return dropped

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"image": load_image(PROCESSED_DIR / r["image"], self.max_side), "text": r["text"]}


class TokenBudgetBatchSampler:
    """Variable-size batches packed up to `max_tokens` (counting padding: longest x count).
    News pages (~1.5-2.5k tokens) go 2 per batch, synthetic lines (~100) up to `max_batch`.
    Batches are built once from length-sorted chunks of a shuffled order (stable len() for the
    Trainer); with shuffle=True the batch ORDER is reshuffled every epoch."""

    def __init__(self, lengths: list[int], max_tokens: int, max_batch: int = 32,
                 shuffle: bool = True, seed: int = 42, chunk: int = 4000):
        rng = random.Random(seed)
        order = list(range(len(lengths)))
        if shuffle:
            rng.shuffle(order)
        self.batches = []
        for c in range(0, len(order), chunk):
            idx = sorted(order[c:c + chunk], key=lengths.__getitem__)
            cur, longest = [], 0
            for i in idx:
                l2 = max(longest, lengths[i])
                if cur and (l2 * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
                    self.batches.append(cur)
                    cur, l2 = [], lengths[i]
                cur.append(i)
                longest = l2
            if cur:
                self.batches.append(cur)
        self.shuffle, self.rng = shuffle, rng

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)


class OCRCollator:
    def __init__(self, processor, max_length: int, instruction: str | None = None):
        self.processor = processor
        self.instruction = instruction
        self.max_length = max_length
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.assistant_ids = tok.encode("assistant\n", add_special_tokens=False)

    def _answer_start(self, ids: list[int]) -> int:
        n = len(self.assistant_ids)
        for pos in range(len(ids) - n - 1, -1, -1):
            if ids[pos] == self.im_start and ids[pos + 1: pos + 1 + n] == self.assistant_ids:
                return pos + 1 + n
        raise ValueError("assistant turn not found")

    def __call__(self, batch: list[dict]) -> dict:
        texts = [self.processor.apply_chat_template(build_messages(ex["image"], ex["text"], **(
            {"instruction": self.instruction} if self.instruction else {})),
                                                    tokenize=False, add_generation_prompt=False)
                 for ex in batch]
        enc = self.processor(text=texts, images=[ex["image"] for ex in batch],
                             return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        for i in range(labels.shape[0]):
            labels[i, : self._answer_start(enc["input_ids"][i].tolist())] = -100
        labels[enc["attention_mask"] == 0] = -100
        if enc["input_ids"].shape[1] > self.max_length:
            print(f"warning: batch length {enc['input_ids'].shape[1]} > max_length {self.max_length}")
        enc["labels"] = labels
        return enc
