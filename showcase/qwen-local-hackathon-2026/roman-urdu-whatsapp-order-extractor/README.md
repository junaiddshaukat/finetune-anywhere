# Roman Urdu WhatsApp Order Extractor

## What it does

Converts messy Roman Urdu and Urdu-English WhatsApp order messages (for example, `"Aoa bhai 3 black fleece hoodies medium size Lahore Model Town bhej dein COD par"`) into a clean, strict JSON object with five fields: `item`, `quantity`, `size`, `city`, and `payment_method` (`"COD"`, `"online"`, or `null`). It includes a lightweight local web UI (`app.py`) backed by Ollama so small online sellers can paste customer messages and extract structured orders 100% offline.

## Why we built it

Small Instagram and WhatsApp businesses in Pakistan receive hundreds of informal Roman Urdu order messages daily, mixing English, Urdu numbers (`ek`, `do`, `teen`, `darjan`), local payment wallets (`JazzCash`, `Easypaisa`, `SadaPay`, `NayaPay`), and Pakistani city names. On our 15 held-out test messages, the base `Qwen3-1.7B` model achieved **0/15** exact matches — wrapping output in markdown fences, pulling greetings (`"Aoa bhai..."`) into the `item` field, misclassifying sector/payment names (`"F-11"`, `"NayaPay"`) as `size`, and using `"null"` strings instead of `null`. Fine-tuning fixed the schema and Roman Urdu understanding.

## Dataset

- **Source & License:** 110 original synthetic examples authored for this project (MIT License).
- **Split:** 80 `train.jsonl`, 15 `valid.jsonl`, 15 `eval.jsonl` (held-out unseen prompts).
- **Validation:** Verified 110 unique prompts with 0 duplicates and 0 cross-split leakage.
- **Coverage:** Clothing, electronics, footwear, accessories, cancellations (`quantity: 0`), quantity corrections, and pre-order inquiries (`quantity: null`).

## Training

- **Tool & Platform:** Unsloth (`2026.11.12`) on Google Colab Free Tesla T4 GPU (15.6 GB VRAM), non-thinking mode (`enable_thinking=False`).
- **Base Model:** `Qwen/Qwen3-1.7B` (revision `a972f4304f86a0a705d35b378d7a7675d58063a7`).
- **Method & Hyperparameters:** QLoRA (`rank=16`, `alpha=32`, `learning_rate=2e-4`, `epochs=3`, `max_seq_length=1024`, `batch_size=1`, `gradient_accumulation_steps=4`, `seed=42`).
- **Software:** Python `3.13.16`, PyTorch `2.14.1+cu130`, Transformers `4.57.6`, TRL `0.22.2`, PEFT `0.21.2`, Datasets `4.3.0`.
- **Duration & Memory:** `128.99s` training time, `4.51 GB` peak allocated GPU memory, `492.29s` GGUF (`Q4_K_M`) export.

## Run it locally

- **Runtime & Hardware:** Ollama `0.40.18` on Windows PC with `20 GB RAM` (CPU inference), `Q4_K_M` GGUF (`1.11 GB`), `num_ctx: 2048`, `temperature: 0`.
- **System Prompt:** `Extract the WhatsApp order into a single JSON object with keys: "item" (string or null), "quantity" (integer or null), "size" (string or null), "city" (string or null), "payment_method" ("COD" | "online" | null). Return only valid JSON.`

Download [`Qwen3-1.7B.Q4_K_M.gguf`](https://huggingface.co/mabdullahramday56/qwen3-1.7b-roman-urdu-whatsapp-order-gguf) and create a `Modelfile`:

```dockerfile
FROM ./Qwen3-1.7B.Q4_K_M.gguf
PARAMETER temperature 0
PARAMETER num_ctx 2048
```

```bash
ollama create whatsapp-order-qwen -f Modelfile
python app.py
# Open http://127.0.0.1:8000 in your browser
```

## Before and after

Across all 15 held-out examples (`eval.jsonl`), exact JSON match improved from **0/15 (Base)** to **8/15 (Fine-tuned)**.

**Prompt:** `Aoa bhai 3 black fleece hoodies medium size Lahore Model Town bhej dein COD par`

- **Base (`Qwen3-1.7B`):**
  ```json
  {"item": "Aoa bhai 3 black fleece hoodies", "quantity": 1, "size": "medium", "city": "Lahore", "payment_method": "COD"}
  ```
- **Fine-tuned (`Qwen3-1.7B` QLoRA):**
  ```json
  {"item": "black fleece hoodies", "quantity": 3, "size": "M", "city": "Lahore", "payment_method": "COD"}
  ```

**Remaining limitations:** On unseen idioms like `"ek darjan"` (dozen), the fine-tuned model predicted `10` instead of `12`, and on an order cancellation without an explicit `0` it output `quantity: null` instead of `0`.

## What we learned

- Even 80 clean, diverse chat examples were enough to eliminate markdown fences and teach Roman Urdu payment/city/size normalization on `Qwen3-1.7B`.
- Quantizing to `Q4_K_M` (`1.11 GB`) allowed smooth offline execution on a CPU-only Windows machine with 20 GB RAM.

## Team

- Muhammad Abdullah ([@Muhammad08-dot](https://github.com/Muhammad08-dot))
- M Taha Tabassum ([@tahatabassum](https://github.com/tahatabassum))
- Built with [finetune-anywhere](https://github.com/junaiddshaukat/finetune-anywhere), [Unsloth](https://unsloth.ai/), [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B), and [Ollama](https://ollama.com/).

