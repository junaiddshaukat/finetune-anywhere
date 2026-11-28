# Manual download (slow or unstable internet)

If `python -m uocr.data.download` stalls, download these files with a browser or a download manager (IDM etc. can resume), then put each one at the path shown. `python -m uocr.data.build` picks them up automatically.

## Minimum set (~430 MB): enough for a first model

| Save as | Download URL | Size |
|---|---|---|
| `data/raw/manual/urdu-ocr-1M/nastaliq/train-00000-of-00010.parquet` | https://huggingface.co/datasets/PuristanLabs1/urdu-ocr-1M/resolve/main/nastaliq/train-00000-of-00010.parquet | 262 MB |
| `data/raw/manual/urdu-ocr-1M/nastaliq/val-00000-of-00001.parquet` | https://huggingface.co/datasets/PuristanLabs1/urdu-ocr-1M/resolve/main/nastaliq/val-00000-of-00001.parquet | 137 MB |
| `data/raw/manual/qaari-0.1-ocr-urdu-news-dataset-small/data/train-00000-of-00001.parquet` | https://huggingface.co/datasets/oddadmix/qaari-0.1-ocr-urdu-news-dataset-small/resolve/main/data/train-00000-of-00001.parquet | 33 MB |
| `data/raw/manual/qaari-0.1-ocr-urdu-news-dataset-small/data/validation-00000-of-00001.parquet` | https://huggingface.co/datasets/oddadmix/qaari-0.1-ocr-urdu-news-dataset-small/resolve/main/data/validation-00000-of-00001.parquet | 4 MB |
| `data/raw/manual/qaari-0.1-ocr-urdu-news-dataset-small/data/test-00000-of-00001.parquet` | https://huggingface.co/datasets/oddadmix/qaari-0.1-ocr-urdu-news-dataset-small/resolve/main/data/test-00000-of-00001.parquet | 4 MB |

## 1908 book (real-world test, 20 MB)

Download the whole folder from https://huggingface.co/datasets/vettedlance/urdu-nastaliq-1908-eval-sample/tree/main, or run `python -m uocr.data.download --only book_1908` on a faster network. It goes to `data/raw/book_1908/` (`lines/*.png` + `lines/metadata.jsonl`).

## Real camera photos: Cursive-Text (CC BY 4.0), important

The only real-photo Urdu set whose license allows training: ~2,500 camera images and 14,100 cropped words.
Download from https://data.mendeley.com/datasets/k5fz57zd9z/3 ("Download All"), unzip into `data/raw/cursive_text/`, and tell me. I'll write the importer once I see its folder/label format.

## Optional: more training data

`nastaliq/train-00001-of-00010.parquet` … `train-00009-of-00010.parquet`, same URL pattern (~260 MB each, ~50k lines each).

## Optional: handwriting (1.9 GB)

Run `python -m uocr.data.download --only handwriting` on a fast network.
