# Qwen3 Fake News Detector

A small Qwen3 model that reads an English news article and answers with one word: **REAL** or **FAKE**. It was fine-tuned and runs fully offline on a MacBook Air M2.

| | Base Qwen3-1.7B (4-bit) | Fine-tuned (this model) |
|---|---|---|
| Accuracy on 200 held-out articles | **66.0%** | **98.5%** |
| Real articles labelled REAL | 36.0% | 98.0% |
| Fake articles labelled FAKE | 96.0% | 99.0% |
| Answers that were not REAL/FAKE | 0.0% | 0.0% |

- **Model:** https://huggingface.co/faisal-001/qwen3-1.7b-fakenews-mlx

## What it does

You paste a news article (title and text). The model replies with a single label, `REAL` or `FAKE`. It's meant as a quick first-pass screen for people learning about misinformation. It runs on a laptop with no internet connection and no cloud API.

## Why we built it

Fake news detection usually depends on cloud services, which means sending what you read to someone else's server. We wanted a detector that is small enough to run privately on an ordinary laptop.

The base Qwen3-1.7B model can follow the instruction, but it is strongly biased: it called **64% of real news articles "FAKE"**. Fine-tuning was needed to teach the model what this kind of real and fake reporting looks like, and to make it answer reliably in the one-word format.

## Dataset

- **Source:** [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) on Hugging Face. It holds about 40k English articles, mostly US politics from 2016–2017, labelled 1 = real and 0 = fake. See the dataset card for its license and origin.
- **Sample used:** 1,300 articles, balanced 50% real and 50% fake, with seed 42.
  - **Train:** 1,000 articles from the dataset's train split
  - **Validation:** 100 articles from its validation split
  - **Held-out test:** 200 articles from its test split, never seen during training
- **Cleaning:**
  - Most real articles start with a byline like `WASHINGTON (Reuters) -`. We **removed that prefix**, otherwise the model could score well just by spotting the word "Reuters".
  - Whitespace was collapsed and each article was truncated to 1,500 characters.
  - Articles shorter than 200 characters were dropped.
- **Format:** MLX-LM chat JSONL, one system message, one user message and one assistant message per article:
  - system: `You are a fake news detector. Read the news article and answer with exactly one word: REAL or FAKE.`
  - user: `Title: <title>\n\n<article text>`
  - assistant: `REAL` or `FAKE`
- **Review:** the labels come from the dataset and were not re-checked by hand. Spot checks found some label noise (see Limitations).

## Training

| Setting | Value |
|---|---|
| Tool | MLX-LM 0.32.2 (`mlx_lm.lora`), MLX 0.32.5 |
| Platform / hardware | Apple Silicon, MacBook Air M2, YOUR_RAM GB unified memory |
| Base model | `mlx-community/Qwen3-1.7B-4bit` (4-bit MLX conversion of `Qwen/Qwen3-1.7B`) |
| Method | LoRA on the 4-bit base (QLoRA), with only the assistant answer trained (`--mask-prompt`) |
| LoRA | rank 8, scale 20, dropout 0, on the last 8 of 28 transformer layers |
| Trainable parameters | ≈2.49M of 1.72B (≈0.15%); the base weights stay frozen |
| Iterations / batch | 300 iterations, batch size 4 (= 1,200 examples, about 1.2 epochs) |
| Learning rate / optimizer | 1e-4, Adam, no schedule |
| Max sequence length | 1,024 tokens, with gradient checkpointing |
| Training time | about 15 minutes per 100 iterations on the M2 (≈45 minutes total) |

```bash
python prepare_data.py   # builds data/train.jsonl, valid.jsonl, test.jsonl

mlx_lm.lora --model mlx-community/Qwen3-1.7B-4bit --train --data data \
  --iters 300 --batch-size 4 --num-layers 8 --learning-rate 1e-4 \
  --max-seq-length 1024 --mask-prompt --grad-checkpoint \
  --steps-per-eval 50 --adapter-path adapters

mlx_lm.fuse --model mlx-community/Qwen3-1.7B-4bit \
  --adapter-path adapters --save-path qwen3-fakenews
```

## Run it locally

You need a Mac with Apple Silicon (M1 or newer, 8 GB RAM is enough). The fused model is about 1 GB, 4-bit.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("faisal-001/qwen3-1.7b-fakenews-mlx")  # downloads once, then works offline

SYSTEM = "You are a fake news detector. Read the news article and answer with exactly one word: REAL or FAKE."
article = "Title: <headline>\n\n<article text, up to ~1,500 characters>"

prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": article}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=8))  # -> REAL or FAKE
```

- **Runtime:** MLX on Apple Silicon
- **Quantization:** 4-bit, group size 64
- **Context used:** up to 1,024 tokens
- **Thinking mode:** off (`enable_thinking=False`)

## Before and after

We used the **same 200 held-out test articles and the same prompt** for the base model and the fine-tuned model. A response counts as correct when the first `REAL`/`FAKE` it contains matches the dataset label. Generation is greedy, with up to 8 new tokens.

**Example: held-out test article #3 (label: REAL).** The prompt was the system message above plus:

> Title: U.S. approves possible $15 billion sale of THAAD missiles to Saudi Arabia
>
> The U.S. State Department has approved the possible sale of a THAAD anti-missile defense system to Saudi Arabia at an estimated cost of $15 billion, the Pentagon said on Friday, citing Iran among regional threats. … *(1,500 characters in total; the full prompt is in `model.yaml`)*

- Base model output: `FAKE` ❌
- Fine-tuned output: `REAL` ✅

**Two more held-out examples (both REAL) that the base model got wrong:**

| Held-out article | Base | Fine-tuned |
|---|---|---|
| #4 "Malaysia's royals call for religious tolerance in rare public intervention" | `FAKE` ❌ | `REAL` ✅ |
| #5 "Illegal miners in South Africa swallow gold in condoms" | `FAKE` ❌ | `REAL` ✅ |

The fine-tuned model fixed 66 of the base model's mistakes. It got 3 of the 200 articles wrong:

| Held-out article | Label | Fine-tuned | Why it probably failed |
|---|---|---|---|
| #48 "Trump on Twitter (Dec 26) - Hillary Clinton, Tax Cut Bill" | REAL | `FAKE` | It's a real Reuters item, but the content is quoted tweets in a casual style, which looks like the fake examples. |
| #56 "reasons why you should apply for a job in the trump administration" | REAL | `FAKE` | The text is lowercase, rambling forum-style writing. This looks like a **labelling error in the dataset**, so the model's answer is arguably right. |
| #165 "longtime clinton aid if america read all of hillarys emails…" | FAKE | `REAL` | The page text copies a real-looking news story about the Trump transition. |

## Limitations

- **This is not a fact-checker.** The model never checks claims against sources. It learned the *writing style* of real and fake articles in this dataset: tone, all-caps headlines, wire-service phrasing.
- **The data is narrow.** It covers English, mostly US political news from 2016–2017. Expect much lower accuracy on recent news, other topics, other countries, other languages, or well-written misinformation.
- **The 98.5% only applies to this dataset.** The test articles come from the same sources as the training data, so the score is not a real-world accuracy figure.
- **It only gives a label.** Use it as a first-pass signal, never as the final word on whether something is true.

## What we learned

- **Look at the base model before training.** It wasn't random: it said "FAKE" for almost everything, which got 96% on fake news but only 36% on real news. Breaking accuracy down by class showed this, where the overall 66% didn't.
- **Remove shortcuts from the data.** Real articles in this dataset nearly all start with "(Reuters)". We stripped that byline so the model had to judge the content, not one word.
- **A small, cheap fine-tune was enough.** 1,000 examples, about 0.15% of the parameters and under an hour on a fanless laptop took accuracy from 66% to 98.5%.
- **Turn off thinking mode.** Qwen3 "thinks" by default. Using `enable_thinking=False` for both training and inference kept answers to a single, fast token.
- **Uncertain:** how well the model handles news outside 2016–2017. That would need a fresh, independently labelled test set.

## Team

- Muhammad Faisal Zahoor ([faisalzahoor1](https://github.com/faisalzahoor1))

**Credits:**
- [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) by the Qwen team (Apache-2.0)
- [mlx-community/Qwen3-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit)
- [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news)
- [MLX-LM](https://github.com/ml-explore/mlx-lm) by Apple
- [finetune-anywhere](https://github.com/junaiddshaukat/finetune-anywhere) by M Junaid
