# onepass

A tiny decision model. You give it a piece of text and typed questions: yes/no, choose one, or pick a level. It returns a probability for every option of every question, all at once, from one forward pass. It generates no tokens, so the JSON it returns is always complete and valid.

- Model: [samadpls/onepass-qwen3-0.6b-GGUF](https://huggingface.co/samadpls/onepass-qwen3-0.6b-GGUF) (all formats and the 1.7B version: [collection](https://huggingface.co/collections/samadpls/onepass-6b06dee3eabd91e4de4df427))
- Code, data pipeline and training notebook: [samadpls/onepass](https://github.com/samadpls/onepass)
- Package: `pip install "onepassai[serve]"` (imports as `onepass`)
- Demo video (1:30): [youtube.com/watch?v=ZZR8TRsX1h4](https://www.youtube.com/watch?v=ZZR8TRsX1h4)

## What it does

Support teams and AI agents make many small decisions: does the customer want a refund, which team should take this, which tool should the agent call next. onepass takes the text (plain or JSON) and the questions, and answers every question with probabilities, for example `refund: true 96%`, `team: billing 98%`.

It is Qwen3-0.6B with a LoRA adapter. The answer to each question is read from the model's scores for the option letters at that question's `Answer:` slot. All questions share one forward pass, and a fixed placeholder after each slot keeps one answer from leaking into the next.

## Why we built it

Letting a chat model write its answer as JSON is slow, costs output tokens, and sometimes breaks. We asked base Qwen3-0.6B (8-bit GGUF, llama.cpp, thinking off) to write a JSON answer for the 400 benchmark test cases. In 37 of them (9.3%) it left out a question. It took 49 generated tokens and about 497 ms per case. onepass reads the same answers from one pass in about 156 ms on the same MacBook Pro GPU, with zero generated tokens and a probability for every option.

Fine-tuning was needed because the base model reads the slots badly: it scores 31% on the benchmark, below a baseline that ignores the input (48%).

## Dataset

68,061 texts with 138,928 questions, built by [src/onepass/build.py](https://github.com/samadpls/onepass/blob/main/src/onepass/build.py). No examples were written by hand or by an LLM for this project.

| Part | Texts | Source and license |
|---|---|---|
| 16 human-labelled datasets | 40,000 (sampled from 86,871) | BoolQ, UNLI, MultiNLI, WANLI, Civil Comments, PAWS, VitaminC, SMS Spam, Banking77, CLINC150, DBpedia, the Davidson hate-speech set, GoEmotions, Measuring Hate Speech, HelpSteer2, BRIGHTER |
| n4ze3m/typed-decisions-synth | 6,523 | MIT; labels from DeepSeek V4.1 Flash |
| decisionmodels/sn92-specialists | 12,730 | CC BY 4.0; open-weight models and exact rules |
| tasksource/procedural-typed-decisions | 8,808 | Apache-2.0; answers from exact rules |

- **Targets are probabilities.** Where several people rated a text, the target is the share of raters who chose each answer (2 of 3 raters gives 0.67).
- **Review.** About 20 candidate typed-question datasets were checked for license, who made the labels, and overlap with the benchmark (exact matches, any shared 10-word run, identical questions). Three passed. Datasets with unclear licenses or labels from closed models were dropped. Options are shuffled, because the right answer came first 51% of the time in generated data.
- **Split.** 2% of each source is held out for validation. Evaluation uses data never trained on: the 400-case test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) and four held-out human-labelled sets. The benchmark's own training split was never used.

## Training

- **Base:** Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Method:** LoRA, rank 32, alpha 64, no dropout, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` in all 28 blocks (20M trainable weights). The loss is cross-entropy only at the answer slots, against the target probabilities.
- **Hyperparameters:** one epoch (3,109 steps of 8,192 tokens), learning rate 2e-4 with a cosine schedule, fp16 with a gradient scaler. One calibration temperature (1.0) fitted on our validation set.
- **Tool and platform:** our own PyTorch loop ([train.py](https://github.com/samadpls/onepass/blob/main/src/onepass/train.py)) with transformers and PEFT 0.22.0, PyTorch 2.11.0, on Kaggle's free 2x Tesla T4. Notebook: [kaggle/train-v2/train.ipynb](https://github.com/samadpls/onepass/blob/main/kaggle/train-v2/train.ipynb).
- **Measured time:** 4.5 hours for the 0.6B. A 1.7B version trained side by side in 8.8 hours.
- **Export:** LoRA merged, converted with llama.cpp to GGUF f16, quantized to Q8_0 and Q4_K_M.

## Run it locally

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uvx --from huggingface_hub hf download samadpls/onepass-qwen3-0.6b-GGUF onepass-qwen3-0.6b-Q8_0.gguf --local-dir .
uvx --from "onepassai[serve]" onepass-serve onepass-qwen3-0.6b-Q8_0.gguf
```

Open http://127.0.0.1:8000 for the demo page, or call the API:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -d '{
  "state": "My card was charged twice for one order, please refund the extra charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "The customer wants a refund.", "criteria": {"true": "Yes.", "false": "No."}},
    "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments and refunds.", "tech": "Bugs and setup."}}
  }
}'
```

- **Runtime:** llama.cpp through the `onepass` runtime (one decode call, scores read only at the answer slots).
- **Quantization:** Q8_0, 639 MB. Q4_K_M (397 MB) keeps the benchmark score but moves single answers by up to 0.22.
- **Memory:** about 2 GB of RAM at peak on CPU (measured), with the default 4,096-token context.
- **Speed:** about 0.5 s per request on the MacBook Pro (M3 Pro) CPU in the demo app; about 0.18 s per benchmark case on its GPU.
- **System prompt:** `You are a decision model. Read the state, then answer every question with the letter of one option.` The prompt layout must match training; [docs/usage.md](https://github.com/samadpls/onepass/blob/main/docs/usage.md) has a copy-paste version for plain transformers.

## Before and after

The same held-out input, base Qwen3-0.6B versus onepass (probabilities read the same way for both):

> State: `{"order_id": "A-1042", "status": "delivered", "days_since_delivery": 41, "return_window_days": 30, "item": "headphones", "request": "return, the item stopped working"}`
> Question (true or false): The order can still be returned under the return window.

| Model | Answer | Judged |
|---|---|---|
| Qwen3-0.6B, before training | true 94% | Wrong: 41 days is past the 30-day window |
| onepass 0.6B, after training | false 69% | Right |

On the full benchmark (400 held-out cases, 2,000 decisions, scored with the benchmark's own formulas):

| Model | Accuracy | KL from gold | Brier | ECE |
|---|---|---|---|---|
| Qwen3-0.6B, before training | 0.311 | 0.888 | 0.428 | 0.306 |
| Label frequencies (ignores the input) | 0.481 | 0.327 | 0.181 | 0.036 |
| onepass 0.6B | 0.601 | 0.270 | 0.154 | 0.049 |
| onepass 1.7B | 0.646 | 0.236 | 0.129 | 0.076 |

The best small general model (under 1B, never trained on the benchmark) on the benchmark's leaderboard scored 0.531.

**A limitation.** On four held-out human-labelled sets, the probabilities improve on all four (KL from gold falls), but accuracy is mixed. On SummEval (summary quality) the 0.6B drops from 0.504 to 0.265: it answers "good" almost every time where experts said "excellent". The top answer on the benchmark is still wrong about 2 times in 5, so read the probabilities, not only the top label.

## What we learned

- The first full model, trained only on the human-labelled sets, scored 0.429: worse than ignoring the input. Those sets are plain sentences; the benchmark is business records in JSON. Adding the three vetted typed-question sets fixed it (0.601).
- Gradient checkpointing was silently off because `from_pretrained` returns a model in eval mode; one `model.train()` cut memory from over 21 GB to 5.3 GB on the Mac.
- 8-bit is nearly free (average probability shift 0.004). 4-bit keeps the score but moves single answers a lot. On the Mac GPU, smaller files were not faster for prompts this short.
- Still uncertain: how well the probabilities hold up on domains far from the training data, as SummEval shows.

## Team

- Abdul Samad ([samadpls](https://github.com/samadpls), [samadpls.com](https://samadpls.com))

Credits: the Qwen team for Qwen3; the authors of the 19 datasets above; [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) for the benchmark; llama.cpp, transformers and PEFT; Kaggle for the free GPUs.
