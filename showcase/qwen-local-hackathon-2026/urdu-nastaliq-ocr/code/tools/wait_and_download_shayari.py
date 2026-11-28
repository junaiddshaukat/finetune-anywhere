"""Poll until access to the gated Urdu Shayari page dataset is granted, then download the test shard
and the first two train shards (~2.2 GB). Writes outputs/logs/shayari_ready when done."""
import os, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import hf_hub_download

REPO = "Ali000007/Urdu-Ocr-Dataset"
FILES = ["data/test-00000-of-00001.parquet", "data/train-00000-of-00006.parquet", "data/train-00001-of-00006.parquet"]
while True:
    try:
        for f in FILES:
            print(time.strftime("%H:%M"), "downloading", f, flush=True)
            hf_hub_download(REPO, f, repo_type="dataset")
        open("outputs/logs/shayari_ready", "w").write("ok")
        print("READY", flush=True)
        break
    except Exception as e:
        print(time.strftime("%H:%M"), "waiting:", str(e)[:90].replace("\n", " "), flush=True)
        time.sleep(600)
