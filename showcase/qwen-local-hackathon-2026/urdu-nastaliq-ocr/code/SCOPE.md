# Scope: Urdu Reader (Nastaliq OCR)

_Last updated: 2026-11-22 · Deadline: 2 days_

## Purpose

Turn photos of **printed Urdu** into accurate, editable text with a **small fine-tuned Qwen3-VL-2B** that can run locally.

**Real-life impact:** a huge amount of Urdu (books, newspapers, government notices, court papers, school material) exists only as images in Nastaliq, a script most OCR handles poorly. Readable text means:
- **blind and low-vision readers** can have it read aloud,
- **archives and libraries** can make it searchable,
- **ordinary people** can copy or share a notice or document instead of retyping it.

## Fine-tuning claim we will prove

Urdu OCR models already exist. The strongest open one is **Qaari 0.1** (Qwen2-VL-2B LoRA, self-reported 2.9% CER, trained on 10k synthetic lines in 5 fonts). Our claim is **not** "first Urdu OCR". It is:

> On **real phone photos and real printed pages**, our fine-tuned Qwen3-VL-2B has a lower CER than Qaari, Tesseract and base Qwen, all tested on the **same images**.

## Benchmark (same test images for every system)

| System | Role |
|---|---|
| Base Qwen3-VL-2B | Before fine-tuning |
| **Our Qwen3-VL-2B + LoRA** | After fine-tuning |
| **Qaari 0.1** (Qwen2-VL-2B LoRA) | Main existing competitor |
| Tesseract `urd` | Classic OCR control |
| Claude (optional) | Big cloud model |

Test tracks: synthetic lines (sanity check), 1908 book (real old print), **our own phone photos** (most important), handwriting (stretch). Qaari's score on `qaari_news` is flagged as **possibly contaminated** (likely its training data).

Metrics: CER and WER (lenient + strict), exact-line match, missing-output rate, seconds per line.

Skipped for time/license: UTRNet (Python 3.7 stack, non-commercial), UTRSet-Real and Proxima UROCR for training (non-commercial), TrOCR/DeepSeek Urdu checkpoints (undocumented).

## In scope (2 days)

- Printed Nastaliq, single lines and clean single-column pages (simple line splitting).
- Data: urdu-ocr-1M Nastaliq subset (MIT), Qaari news (MIT), OUHD-L handwriting (CC BY 4.0), and **Cursive-Text** real camera images (CC BY 4.0, Mendeley manual download; importer to be added once we see its format).
- Evaluation: CER/WER per source (synthetic, news, 1908 book, handwriting, our photos). Base vs fine-tuned, optionally Claude.
- Website: photo → text, copy, browser read-aloud.
- Training on the RTX 3090. Smoke tests on the 4070 laptop.

## Out of scope

- Multi-column newspaper layouts, tables, mixed image/text pages (needs a layout model).
- Strong handwriting accuracy (only 1.9k training lines; we report it honestly as a stretch result).
- Phone app and offline GGUF build (the export script is ready; build after the hackathon).
- Naskh font, Urdu-English mixed lines, Arabic/Persian.

## Risks

| Risk | Mitigation |
|---|---|
| Base model already reads Nastaliq well (no gain to show) | Run `eval_base` first; if CER is already low, change the focus (handwriting, old books) |
| Synthetic training doesn't transfer to real photos | Mix in real news lines (oversampled); report real-data CER separately; add our own photos |
| Sentence leakage inflates scores | Sentence-level splits for synthetic data |
| Line splitting fails on tilted/complex pages | Demo with clean pages; "one line" mode as fallback |
| Unstable internet slows downloads | Downloader retries; 1 shard (~50k lines) is enough for v1 |
| 1908 book is evaluation-only | Never trained on; test only; noted in the manifest |

## Change log

- 2026-11-22: Added competitor benchmark (Qaari, Tesseract) after landscape research; target claim moved to real photos.
- 2026-11-22: Project created (pivot from the farmer assistant, which needed field validation we can't do in 2 days).
