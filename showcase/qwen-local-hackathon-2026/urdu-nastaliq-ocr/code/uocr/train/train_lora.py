"""LoRA fine-tuning of Qwen3-VL for Urdu OCR.

Usage:
    python -m uocr.train.train_lora --config configs/train_rtx3090_24gb.yaml
    python -m uocr.train.train_lora --config configs/train_laptop_8gb.yaml --max-train 32 --max-steps 2   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Variable batch shapes fragment the CUDA cache (28-30 GB reserved for ~18 GB used); on Windows
# that silently spills into shared system memory. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.stdout.reconfigure(line_buffering=True)  # loss logs show up live when redirected to a file

import torch  # noqa: E402
import yaml  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from uocr.config import ROOT
from uocr.modeling import LM_LORA_TARGET, load_model, load_processor
from uocr.prompts import PROMPTS
from uocr.train.dataset import OCRCollator, OCRDataset, TokenBudgetBatchSampler


class TokenBudgetTrainer(Trainer):
    """News pages are ~1.5-2.5k tokens, synthetic lines ~100. Fixed-size batches either blow the
    24 GB budget (4 pages = 10k tokens of 152k-vocab logits) or waste the GPU on tiny line
    batches, so batches are packed up to a token budget instead (train and eval)."""

    max_tokens: int = 6144
    max_batch: int = 32

    def _budget_loader(self, ds, shuffle: bool):
        sampler = TokenBudgetBatchSampler(ds.lengths, self.max_tokens, self.max_batch, shuffle=shuffle,
                                          seed=self.args.seed)
        n = self.args.dataloader_num_workers
        return self.accelerator.prepare(DataLoader(
            ds, batch_sampler=sampler, collate_fn=self.data_collator, num_workers=n,
            pin_memory=True, persistent_workers=n > 0))

    def get_train_dataloader(self):
        return self._budget_loader(self.train_dataset, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        return self._budget_loader(eval_dataset if eval_dataset is not None else self.eval_dataset, shuffle=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tcfg, lcfg = cfg["train"], cfg["lora"]
    out_dir = ROOT / cfg["output_dir"]

    processor = load_processor(cfg["base_model"])
    model = load_model(cfg["base_model"], quantize_4bit=cfg.get("quantize_4bit", False))
    model.config.use_cache = False
    if cfg.get("quantize_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tcfg["gradient_checkpointing"])

    if cfg.get("init_adapter"):   # continue training an existing LoRA (its r/alpha/targets are kept)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / cfg["init_adapter"]), is_trainable=True)
        print(f"continuing from adapter {cfg['init_adapter']}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"], bias="none",
            target_modules="all-linear" if lcfg["target"] == "all" else LM_LORA_TARGET,
            task_type="CAUSAL_LM"))
    if tcfg["gradient_checkpointing"]:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = OCRDataset("train", cfg["max_side"], limit=args.max_train or tcfg.get("max_train_examples"),
                          sources=cfg.get("train_sources"), mix=cfg.get("source_mix"))
    eval_ds = OCRDataset("val", cfg["max_side"], limit=tcfg.get("max_eval_examples"),
                         sources=cfg.get("train_sources"))
    for name, ds in (("train", train_ds), ("val", eval_ds)):
        n = ds.drop_too_long(processor.tokenizer, cfg["max_length"], px_per_token=cfg.get("image_token_px", 32))
        if n:
            print(f"dropped {n} {name} examples longer than max_length={cfg['max_length']} tokens")
    print(f"train tokens (est.): {sum(train_ds.lengths) / 1e6:.1f}M")
    print(f"train={len(train_ds)} val={len(eval_ds)}")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=tcfg["learning_rate"],
        warmup_steps=tcfg["warmup_ratio"],  # transformers 5: a float in [0, 1) is a ratio
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        logging_steps=tcfg["logging_steps"],
        dataloader_num_workers=tcfg["num_workers"],
        remove_unused_columns=False,
        report_to="none",
    )
    TokenBudgetTrainer.max_tokens = tcfg.get("max_tokens_per_batch", 6144)
    TokenBudgetTrainer.max_batch = tcfg.get("max_batch_examples", 32)
    trainer = TokenBudgetTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=OCRCollator(processor, cfg["max_length"], PROMPTS[cfg.get("prompt", "default")]))
    trainer.train(resume_from_checkpoint=args.resume or None)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    shutil.copy(config_path, out_dir / "train_config.yaml")
    (out_dir / "train_metrics.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved")
    print(f"Adapter saved -> {out_dir}")


if __name__ == "__main__":
    main()
