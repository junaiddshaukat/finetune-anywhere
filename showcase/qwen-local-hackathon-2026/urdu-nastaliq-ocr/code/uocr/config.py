"""Shared paths and environment loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
PREVIEW_DIR = DATA_DIR / "preview"
MY_PHOTOS_DIR = DATA_DIR / "my_photos"
OUTPUTS_DIR = ROOT / "outputs"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"

BASE_MODEL = os.getenv("UOCR_BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or None

# Text-line images are resized so the long side is at most this many pixels.
# Urdu lines are wide and short, so this mostly limits width. Same value for train + inference.
DEFAULT_MAX_SIDE = 1024

for d in (RAW_DIR, PROCESSED_DIR, IMAGES_DIR, OUTPUTS_DIR, MY_PHOTOS_DIR):
    d.mkdir(parents=True, exist_ok=True)
