# Urdu Nastaliq OCR: fine-tuned Qwen3-VL-2B that runs offline

Photograph printed or handwritten Urdu (a book, newspaper, notice, sign or notebook page) and get the text back, so you can copy it, search it or have it read aloud. Most OCR tools read Nastaliq's stacked, joined letters poorly. This project fine-tunes a small **Qwen3-VL-2B** with LoRA to do this one job, measures the gain with character error rate (CER), and exports the model to **GGUF** so it runs on a CPU with no internet and no API key.

See [SCOPE.md](SCOPE.md) for goals and limits, and [submission/README.md](submission/README.md) for the full write-up.

## Results

CER / WER in percent (lower is better), on the 362 test items every system was run on:

| Test set (items) | Tesseract | Base Qwen3-VL-2B | Qaari 0.1 | Ours v1 | **Ours v3** |
|---|---:|---:|---:|---:|---:|
| Handwriting, unseen writers (200) | 61.4 / 96.7 | 41.5 / 77.1 | 241.3 / 282.1 | **8.7** / 27.3 | 9.2 / **27.7** |
| News pages (87) | 11.1 / 17.1 | 86.2 / 96.5 | 6.9 / 8.9 \* | **5.7** / **6.4** | **5.7** / 6.5 |
| Printed lines, synthetic (24) | 59.7 / 90.4 | 45.8 / 79.0 | 71.2 / 107.7 | **2.8** / 20.6 | 3.1 / **18.8** |
| 1908 book, never trained on (51) | 67.1 / 100.9 | 85.7 / 114.1 | 194.7 / 217.1 | 80.1 / 111.0 | **34.1 / 73.2** |
| **All 362** | 18.7 / 29.4 | 82.8 / 96.3 | 36.0 / 43.7 | 11.0 / 15.6 | **7.9 / 13.0** |

\* Qaari was trained on this news dataset, so its score there is not a held-out result. Our runs use the letter-size enlargement described below; the baselines did not. Caveats are in [submission/README.md](submission/README.md#results).

## How it works

- **Three LoRA rounds** on one RTX 3090 (r=32, alpha=64, all linear layers, bf16):
  - **v1**: synthetic Nastaliq lines, news pages and tablet handwriting (82,670 examples).
  - **v2**: the same data re-rendered as photos (notebook paper, multi-line pages).
  - **v3**: a second typeface (Noto Nastaliq Urdu).
- **Letter-size enlargement.** Qwen-VL reads images in 32-pixel patches, so small Nastaliq letters get lost. Images are enlarged until letters are about 48 px tall, measured from the ink blobs. On one typeset page this took CER from 163% (a repeating loop) to about 12%, with no training.
- **Solid-ink training data.** Eroded synthetic lines taught the model to guess words for letters it could not see, so the later rounds train only on solid-ink lines.
- **No leakage.** Printed data is split by sentence and handwriting by writer. The 1908 book and our own photos are test-only.

## Repository layout

```text
urdu_ocr/
├── pipeline.py              # one command: download → build → preview → eval → train → eval → compare
├── Makefile                 # demo / app / llama-server shortcuts (`make help`)
├── app/                     # web app: photo → Urdu text (copy, read aloud, side-by-side models)
├── configs/                 # training configs (3090 rounds v1–v3, laptop smoke test)
├── data/                    # downloads and processed data (git-ignored), see MANUAL_DOWNLOAD.md
├── submission/              # hackathon write-up and model card
├── tools/                   # eval helpers, GGUF eval, LoRA merge, font rendering, status
└── uocr/
    ├── text.py              # Urdu normalization + CER/WER
    ├── prompts.py           # the OCR instruction
    ├── modeling.py          # load Qwen3-VL (+ LoRA)
    ├── pages.py             # line splitting and letter-size enlargement
    ├── inference.py         # UrduReader.read_lines() / read_page()
    ├── data/                # download, build splits, HTML preview
    ├── train/               # LoRA fine-tuning
    ├── eval/                # CER/WER per source, HTML reports
    └── export/              # merge LoRA → GGUF
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

`.env` holds an optional `HF_TOKEN` and `ANTHROPIC_API_KEY` (only for the Claude baseline). It is git-ignored; never commit it.

## Run

```bash
# Full pipeline on a 24 GB GPU
python pipeline.py --config configs/train_rtx3090_24gb.yaml --handwriting

# Quick plumbing test on a laptop (minutes)
python pipeline.py --config configs/train_laptop_8gb.yaml --smoke

# Step by step
python -m uocr.data.download --handwriting
python -m uocr.data.build
python -m uocr.data.preview                       # open data/preview/index.html
python -m uocr.eval.evaluate --name base --per-source 200
python -m uocr.train.train_lora --config configs/train_rtx3090_24gb.yaml
python -m uocr.eval.evaluate --name finetuned --adapter outputs/qwen3vl2b-urdu-ocr-lora --per-source 200
python -m uocr.eval.evaluate --compare base finetuned
```

Baselines: `--backend qaari`, `--backend tesseract`, `--backend claude`. For Tesseract, install it and add `urd.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best).

### Web app

```bash
make app          # fine-tuned model in PyTorch on the GPU
make llama        # llama-server with the GGUF (CPU), then in another terminal:
make app-local
make demo         # several models side by side
make app-mock     # fake model, for UI work
```

Or directly: `uvicorn app.server:app --host 0.0.0.0 --port 8001`, then open http://localhost:8001.

### Offline (GGUF + llama.cpp)

```bash
llama-server -m urdu-ocr-2b-v3-Q8_0.gguf --mmproj urdu-ocr-2b-v3-mmproj-f16.gguf -c 8192 -ngl 0 --jinja
```

On an i7-13700KF CPU, Q8_0 (1.8 GB) reads a handwritten line in about 3.5 s and a 13-line page in about 53 s, using 5.6 GB RAM peak.

## What is not in this repo

Datasets, processed data, training outputs, LoRA adapters and GGUF weights are git-ignored because of size and licences. Re-create the data with `uocr.data.download`. Model weights are published separately.

## Data and licences

| Source | Licence | Used for |
|---|---|---|
| [PuristanLabs1/urdu-ocr-1M](https://huggingface.co/datasets/PuristanLabs1/urdu-ocr-1M) (Nastaliq) | MIT | synthetic lines |
| [oddadmix Qaari 0.1 news](https://huggingface.co/datasets/oddadmix/qaari-0.1-ocr-urdu-news-dataset-small) | MIT | news pages |
| [OUHD-L handwriting](https://huggingface.co/datasets/saad2002/ouhd-l-online-urdu-nastaliq-handwriting) | CC BY 4.0 | handwriting (split by writer) |
| [Noto Nastaliq Urdu](https://fonts.google.com/noto/specimen/Noto+Nastaliq+Urdu) | SIL OFL 1.1 | second typeface |
| [1908 book sample](https://huggingface.co/datasets/vettedlance/urdu-nastaliq-1908-eval-sample) | evaluation only | test only, never trained on |

Base model: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (Apache 2.0). Fonts bundled in `app/static/fonts/` are under the SIL OFL (licence files included).
