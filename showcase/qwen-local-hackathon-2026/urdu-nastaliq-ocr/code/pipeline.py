"""End-to-end pipeline: download -> build -> preview -> baseline eval -> train -> eval -> compare.

Usage:
    python pipeline.py --config configs/train_rtx3090_24gb.yaml          # everything
    python pipeline.py --config configs/train_laptop_8gb.yaml --smoke     # tiny run to test the plumbing
    python pipeline.py --steps build preview                              # only some steps

Steps: download build preview eval_base train eval_finetuned compare report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ALL_STEPS = ["download", "build", "preview", "eval_base", "train", "eval_finetuned", "compare", "report"]


def run(args: list[str]) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_rtx3090_24gb.yaml")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    ap.add_argument("--synth-shards", type=int, default=1)
    ap.add_argument("--handwriting", action="store_true", help="also download the 1.9 GB handwriting set")
    ap.add_argument("--per-source", type=int, default=200, help="test lines per source for eval")
    ap.add_argument("--smoke", action="store_true", help="tiny train/eval just to test the pipeline")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    adapter = cfg["output_dir"]
    per_source = "8" if args.smoke else str(args.per_source)
    max_side = str(cfg["max_side"])

    for step in args.steps:
        print(f"\n========== {step} ==========")
        if step == "download":
            run(["-m", "uocr.data.download", "--synth-shards", str(args.synth_shards)]
                + (["--handwriting"] if args.handwriting else []))
        elif step == "build":
            run(["-m", "uocr.data.build"] + (["--max-synth-train", "500"] if args.smoke else []))
        elif step == "preview":
            run(["-m", "uocr.data.preview"])
        elif step == "eval_base":
            run(["-m", "uocr.eval.evaluate", "--name", "base", "--per-source", per_source,
                 "--max-side", max_side, "--base-model", cfg["base_model"]])
        elif step == "train":
            run(["-m", "uocr.train.train_lora", "--config", args.config]
                + (["--max-train", "32", "--max-steps", "2"] if args.smoke else []))
        elif step == "eval_finetuned":
            run(["-m", "uocr.eval.evaluate", "--name", "finetuned", "--adapter", adapter,
                 "--per-source", per_source, "--max-side", max_side, "--base-model", cfg["base_model"]])
        elif step == "compare":
            run(["-m", "uocr.eval.evaluate", "--compare", "base", "finetuned"])
        elif step == "report":
            run(["-m", "uocr.eval.report", "--name", "base", "finetuned"])
    print("\nDone. Start the app: uvicorn app.server:app --port 8001")


if __name__ == "__main__":
    main()
