# Receipt to Claim

Code: https://github.com/Huzaifakamran/receipt-to-claim

## What it does

An employee or HR clerk drops photos of medical and expense receipts (phone camera or WhatsApp-forwarded images) into a local web app. `qwen3.5:4b`, running in Ollama on the same laptop, reads each photo and returns the vendor, date, receipt and FBR invoice numbers, patient name, line items, discount, tax, the final total and a category (consultation, medicine, lab test, hospital, fuel, meals, travel, ...). Rule checks flag the rows a person must look at. The user fixes any cell and downloads an Excel claim sheet with flagged rows highlighted, a total, and a by-category summary, plus a CSV.

## Why we built it

In many Pakistani offices a medical reimbursement claim is a Word file of pasted receipt photos, and someone in HR or finance retypes every amount. The receipts are mostly handwritten clinic slips and pharmacy cash memos, often photographed overlapping, and mixed with prescriptions that have no amount on them. They also carry the patient's name, father's name, age, address and treatment, which is a poor fit for a cloud API. A 4B vision model that runs on an ordinary laptop GPU keeps the data on the machine and costs nothing per page.

Fine-tuning was not needed for a first version. Structured JSON output plus a few rule checks got usable results. The remaining errors are handwritten amounts, which is where a fine-tune on labelled Pakistani receipts would help next.

## Dataset

No training data. Two evaluation sets:

- **Author's own medical claims (private, development set).** 12 photos: 7 receipts (3 handwritten pharmacy cash memos, 4 clinic fee slips from Karachi, Islamabad and Rawalpindi) and 5 prescriptions or OPD tokens with no amount. Labelled by hand and confirmed by the author. The images contain personal medical details and are not published. They were also looked at while writing the prompt, so this is a development set, not a held-out test.
- **CORD-v2 test split (public, held out).** [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), CC-BY-4.0. 95 of the 100 test receipts have a printed total; only the total is scored, because CORD masks store names and dates. The prompt was frozen before this set was run once.

## Training

None. `training_tool`, `training_platform` and `method` are `none`. The model is the stock `qwen3.5:4b` from the Ollama library (Q4_K_M, 4.2B parameters plus a 0.33B vision projector, Apache-2.0).

Inference settings: temperature 0, 8K context, thinking off, image resized to at most 1600 px on the long side, and the output constrained by a JSON schema passed as Ollama's `format`. The model copies the total line as written; code converts it to a number.

## Run it locally

Needs [Ollama](https://ollama.com/download) and [uv](https://docs.astral.sh/uv/). Uses about 3.4 GB of VRAM. Tested on an RTX 4070 Laptop GPU (8 GB) with 32 GB RAM on Windows 11, Ollama 0.40.18. No system prompt; the full extraction prompt is in `receipt2claim/extract.py`.

```bash
ollama pull qwen3.5:4b
git clone https://github.com/Huzaifakamran/receipt-to-claim.git
cd receipt-to-claim
uv sync
uv run receipt2claim ui                                   # web app at http://127.0.0.1:7860
uv run python scripts/make_demo_receipts.py data/demo     # 7 fictional receipts to try it on
uv run receipt2claim sheet data/demo --out claim.xlsx     # same pipeline from the command line
```

## Before and after

Not fine-tuned, so these are real inputs and the app's real outputs.

**Real Pakistani medical receipts (development set, 7 receipts + 5 prescriptions):**

| Field | Correct |
| --- | ---: |
| Vendor | 7 / 7 |
| Category | 7 / 7 |
| Date | 6 / 7 |
| Total | 5 / 7 |
| No total invented on a prescription page | 5 / 5 |

**CORD-v2 test (95 receipts, held out):**

| Run | Totals correct |
| --- | ---: |
| Frozen prompt, held out | **84 / 95 (88.4%)** |
| After error analysis: amount conversion moved into code (post-hoc, same set) | 92 / 95 (96.8%) |

The first raw score was 80/95; four of those misses were my own label-conversion bug (totals with cents such as `63,000.00` turned 100x too big), and the model was right. Eight of the remaining misses were the model copying `TOTAL 31.000` correctly and then converting it to 31.0. Parsing the copied text in code fixed that class. That change came after seeing the results, so it is reported separately.

**Speed:** about 5 s per receipt (p95 6.3 s) on the RTX 4070 Laptop GPU, roughly 62 tokens/s, after a one-time 5 s model load. 0 malformed JSON outputs.

Example, a handwritten clinic fee slip with "OPD Charges Rs 700/-" and, lower down, "Other: T - 700/-" (T = total):

- First prompt: total `1400`. The model added the two lines.
- Final prompt, which asks for the total line exactly as written before the number: `total_as_written: "T - 700/-"`, total `700`. Correct.

How I judged: exact match for totals (within 0.5) and dates, normalised fuzzy match for vendor names, exact match for category, all against hand labels.

Failure I did not fix: a highlighted handwritten `Rs 11000/-` on a homeopathic clinic slip was read as `Rs 1600/-`, and no rule flagged it. A plausible wrong amount is the worst kind of error for a claim, so the sheet keeps "Total as written" next to each amount and a person still signs off.

## What we learned

- Constraining the output with a JSON schema in Ollama removed parsing failures completely.
- Making the model copy the evidence (the total line as written) before giving the number helped more than anything else: total accuracy on handwritten receipts went from 3/7 to 5/7, and the reviewer gets the evidence for free.
- A 4B model copies numbers better than it converts them. Letting plain code parse the copied text (`31.000` → 31000, `9.500,00` → 9500) fixed most of the remaining printed-receipt errors.
- Check your labels before blaming the model: 4 of my first 15 benchmark "misses" were my own label-conversion bug.
- Qwen3.5's thinking mode did not pay off here: about 10x slower (49 s per image), 3 of 12 answers ran out of context, and no accuracy gain.
- Prescriptions mixed into a claim are handled well: the model left the total empty on all 5.
- Still uncertain: handwritten amounts, day-first dates with one-digit months (`23/1/26` read as November), Urdu-only receipts (none in the test set), and CPU-only speed (not measured).

## Team

- Muhammad Huzaifa ([@Huzaifakamran](https://github.com/Huzaifakamran)), solo.

Credits: Qwen3.5 by the Qwen team (Apache-2.0) via Ollama; CORD dataset by NAVER Clova (Park et al., 2019, CC-BY-4.0); Gradio, openpyxl, pydantic, httpx.
