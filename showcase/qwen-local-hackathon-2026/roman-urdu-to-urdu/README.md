
# Qwen3-0.6B Roman Urdu → Urdu script

A 0.6B model that turns Roman Urdu, the way people type it on WhatsApp, into proper Urdu script. It runs offline on a laptop: 643 MB in Ollama, about 0.3 s per sentence on a MacBook Air M2.

| Roman Urdu (input) | Urdu script (this model) |
|---|---|
| golgappay khane chalein? | گولگپے کھانے چلیں؟ |
| aap ka naam kya hai | آپ کا نام کیا ہے؟ |
| ye kaam kaun karay ga? | یہ کام کون کرے گا؟ |

Built for the Qwen Local Hackathon 2026 (Qwen Pakistan).

## Why

Most Pakistanis type Urdu in Latin letters, because that's what phone keyboards make easy. But forms, documents, signage and printed material need real Urdu script, and typing it is slow for most people. Rule-based converters struggle because Roman Urdu has no fixed spelling: *hai*, *he* and *hy* are all the same word. A small model can learn those patterns from examples. It can also run locally, so private messages never leave the device.

The base Qwen3-0.6B can't do this task at all. It echoes the Roman text back unchanged (see results).

## Files in this repo

| Path | What it is |
|---|---|
| `model.safetensors` and config files (repo root) | Merged full-precision model: LoRA fused into Qwen3-0.6B. Use with Transformers, or import into Ollama. |
| `mlx-4bit/` | 4-bit MLX version (331 MB) for Apple Silicon. |
| `adapter/` | The LoRA adapter on its own (MLX format). |
| `Modelfile` | Ollama import file (int4). |
| `demo.py` | Command-line demo that calls Ollama. Python standard library only. |
| `scripts/` | Data preparation and evaluation scripts. |
| `results/` | Every model output on both test sets, plus training logs. |

## Run it locally

### Ollama (tested with Ollama 0.40.16 on macOS, Apple M2)

```bash
hf download narjisb99/qwen3-0.6b-roman-urdu-to-urdu --exclude "mlx-4bit/*" --exclude "adapter/*" --local-dir roman-urdu-to-urdu
cd roman-urdu-to-urdu
ollama create roman-urdu-to-urdu -q int4 -f Modelfile
python3 demo.py "aap ka naam kya hai"
python3 demo.py            # interactive mode
```

The int4 model is 643 MB and used about 660 MB of memory while loaded (from `ollama ps`). Context is 4096 tokens, though inputs are single sentences.

System prompt (baked into the Modelfile):

```
Convert the Roman Urdu text into Urdu script. Return only the Urdu text, with no explanation.
```

Qwen3 starts every answer with an empty `<think></think>` block, even with thinking turned off. `demo.py` removes it. Through the API, send `"think": false` and strip the tags yourself.

### MLX (Apple Silicon)

```bash
pip install mlx-lm
hf download narjisb99/qwen3-0.6b-roman-urdu-to-urdu --include "mlx-4bit/*" --local-dir rutu
mlx_lm.generate --model rutu/mlx-4bit --temp 0 \
  --system-prompt "Convert the Roman Urdu text into Urdu script. Return only the Urdu text, with no explanation." \
  --prompt "golgappay khane chalein?"
```

The chat template in `mlx-4bit/` already turns thinking off.

## Results

All numbers come from held-out sentences the model never saw in training, decoded greedily (temperature 0).

- **Exact match:** the whole sentence is character-perfect.
- **CER (character error rate):** Levenshtein edit distance divided by reference length. Lower is better.

Both are measured after Unicode NFC normalisation and whitespace collapsing.

### Everyday sentences (60 test sentences written by the team)

| Model | Exact match | CER |
|---|---:|---:|
| Qwen3-0.6B base (4-bit MLX) | 0% | 107.7% |
| v1: trained on Wikipedia only | 3.3% | 17.8% |
| **v2: Wikipedia + everyday (this model), 4-bit MLX + adapter** | **18.3%** | **10.6%** |
| v2 merged, then 4-bit MLX (`mlx-4bit/`) | 18.3% | 11.0% |
| v2 merged, then int4 in Ollama | 21.7% | 11.0% |

### Wikipedia sentences (300 test sentences from Dakshina)

| Model | Exact match | CER | Seconds per sentence (M2) |
|---|---:|---:|---:|
| Qwen3-0.6B base | 0% | 100.2% | 0.39 |
| v1 | 10.7% | 10.2% | 1.13 |
| v2 (this model) | 9.7% | 11.3% | 0.54 |

Exact match looks low because one wrong letter fails the whole sentence. About 89% of characters are right. For scale: the Dakshina paper reports that human annotators disagreed with each other on roughly 3.5–8.5% of characters. The Wikipedia set was not re-run on the Ollama int4 build. On the everyday set, int4 cost about half a CER point.

### Same prompt, base and fine-tuned (everyday test set)

| Input | Reference | Base | v1 | v2 |
|---|---|---|---|---|
| ye kaam kaun karay ga? | یہ کام کون کرے گا؟ | ye kaam kaun karay ga? | یہ کام کارے گا؟ | یہ کام کون کرے گا؟ ✅ |
| golgappay khane chalein? | گول گپے کھانے چلیں؟ | golgappay khane chalein? | گلوگپائے خانے چلیں؟ | گولگپے کھانے چلیں؟ (missing space only) |
| office ki bus subah saat baje aati hai | آفس کی بس صبح سات بجے آتی ہے۔ | (echoed back) | آفیس کی بس سب سب سات سات بجے ایتی ہے | ایجو کی بس صبح سات بجے آتی ہے۔ |

### Where it still fails

**English loanwords are the main weakness.** Dakshina is Wikipedia text and contains almost no casual English, so words like *discount*, *walk* and *WhatsApp* come out garbled:

| Input | Reference | v2 output |
|---|---|---|
| thora sa discount kar dein bhai | تھوڑا سا ڈسکاؤنٹ کر دیں بھائی۔ | اٹھارا سا ڈسکٹ کر دیں بھائی۔ |
| meeting ka link mujhe whatsapp kar do | میٹنگ کا لنک مجھے واٹس ایپ کر دو۔ | میٹنگ کا لائن مجھے ویٹم ایٹر کر دو۔ |
| rozana thori der walk kiya karo | روزانہ تھوڑی دیر واک کیا کرو۔ | روزنہ طوری ڈیر وکل کیا کرو۔ |

Other limits:

- The everyday test set is only 60 sentences, so differences of a few points (for example, int4 scoring 21.7% vs 18.3%) are within noise.
- It's tuned for single sentences of up to about 30 words. Long paragraphs are untested.
- Quantization cost a little: *kaun* came out as کاون in the Ollama int4 build, though it was correct before quantizing.

## Dataset

**1. Dakshina, Urdu subset** (Google Research, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). These are Urdu Wikipedia sentences that native speakers romanised by hand. We used the copy at [Anvesh-Lankala/Copy_Dakshina_Google_research_dataset](https://huggingface.co/datasets/Anvesh-Lankala/Copy_Dakshina_Google_research_dataset), file `data/ur-00000-of-00001.parquet`.

- Filtered to sentences of 30 words or fewer.
- Removed empty rows and deduplicated on both the Roman and the Urdu side.
- Shuffled with seed 42 and split into **7,892 train / 200 validation / 300 test**.
- See `scripts/prepare_data.py`.

**2. Everyday sentences** (182, written by the two of us for this project). These are short texting-style sentences across 10 categories: greetings, family, office, shopping, travel, food, health, questions, English loanwords, and places and events.

- We wrote them ourselves. We did not copy them from websites or chats, and they contain no personal details.
- We reviewed every row by hand. 4 rows were corrected because the Urdu column had been translated or reordered instead of transliterated (see the `FIXES` list in `scripts/prepare_everyday.py`).
- Shuffled with seed 42 and split into **60 test / 122 train**. No sentence appears in both.
- The 122 training sentences were repeated 5 times, so they carry weight next to 7,892 Wikipedia rows. Final training set: 8,502 rows.

Every row uses the same system prompt and chat format: system prompt, then the Roman Urdu as the user turn, then the Urdu script as the assistant turn.

## Training

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Method | LoRA on a 4-bit (MLX) base, i.e. QLoRA |
| Tool | `mlx-lm` 0.32.2 (`mlx` 0.32.4), using the Apple Silicon path from the finetune-anywhere starter repo |
| Hardware | MacBook Air, Apple M2, 16 GB unified memory |
| LoRA settings | rank 16, scale 2.0, dropout 0, last 16 layers |
| Optimiser | Adam, learning rate 2e-4, batch size 4, max sequence length 256, seed 42 |
| Steps | 3,000, with loss on the assistant turn only (`--mask-prompt`) |
| Training data seen | about 539k tokens |
| Compute time | about 55 minutes, from the MLX step logs |
| Validation loss | 2.713 → 0.293 |

Commands:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-0.6B --mlx-path outputs/base-4bit -q --q-bits 4

mlx_lm.lora --model outputs/base-4bit --train --data data/mixed --iters 3000 \
  --batch-size 4 --num-layers 16 --learning-rate 0.0002 --max-seq-length 256 \
  --mask-prompt --adapter-path outputs/adapters-v2 \
  --config lora.yaml --steps-per-report 100 --steps-per-eval 500 --save-every 500 --val-batches 25
# lora.yaml: lora_parameters {rank: 16, scale: 2.0, dropout: 0.0}, seed: 42

mlx_lm.fuse --model Qwen/Qwen3-0.6B --adapter-path outputs/adapters-v2 --save-path outputs/merged
# then set a non-thinking chat template (configure_chat.py in the starter repo)
mlx_lm.convert --hf-path outputs/merged --mlx-path outputs/mlx-4bit -q --q-bits 4
ollama create roman-urdu-to-urdu -q int4 -f Modelfile

python scripts/evaluate.py --model outputs/base-4bit --adapter outputs/adapters-v2 \
  --eval data/everyday_test.jsonl --out results/everyday-v2.json
```

## What we learned

- **Test data matters most.** v1 looked good on Wikipedia text (10.2% CER) but was weak on how people actually type (17.8%). Writing 182 sentences of our own fixed most of that gap. It cut everyday CER by about 40%, at the cost of about 1 CER point on Wikipedia.
- **Masking the prompt also helped.** v2 trains only on the answer, and that made generation faster (0.54 s vs 1.13 s per sentence), mostly because v1 sometimes rambled with repeated words.
- **4-bit is nearly free here.** The quantized builds score within about half a CER point of the full model.
- **Still open:** English loanwords need many more examples than our 122. That's the next thing to fix.

## Team and credits

Built by two sisters for the Qwen Local Hackathon 2026. Syeda Narjis Batool ([syeda-batool99](https://github.com/syeda-batool99)) built the training and evaluation pipeline. The other sister wrote 150 of the 182 everyday sentences.

- **Base model:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) by the Qwen team, Apache 2.0.
- **Data:** Dakshina dataset, Roark et al., *Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset*, LREC 2020, CC BY-SA 4.0.
- **Tools:** [MLX / mlx-lm](https://github.com/ml-explore/mlx-lm), [Ollama](https://ollama.com), and the finetune-anywhere starter repo from Qwen Pakistan.
