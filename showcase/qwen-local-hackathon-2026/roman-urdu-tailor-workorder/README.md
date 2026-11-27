# Roman Urdu tailor work-order compiler

A Qwen3-1.7B LoRA fine-tune that turns a spoken Roman Urdu alteration instruction into a JSON
work order a shop can act on. Runs entirely offline in Ollama on a 6 GB laptop GPU.

## What it does

The user types what a customer says at the counter, in Roman Urdu:

```
Kameez ka daman do inch chota, bazu ek inch chhoti, aur pocket add karna hai.
```

and gets back a record with a fixed schema - twelve garment parts, seven actions:

```json
{"jobs": [{"part": "hem", "action": "shorten", "amount_inches": 2},
          {"part": "sleeve", "action": "shorten", "amount_inches": 1},
          {"part": "pocket", "action": "add", "amount_inches": null}],
 "cancelled_parts": [], "needs_clarification": false}
```

`cancelled_parts` carries the withdrawal rule that trips people up: in
`Bazu 2 inch chhoti karni thi, rehne dein`, the sleeve edit is *withdrawn*, so the part is
cancelled and no job is emitted for it. `needs_clarification` is the model's only permitted way
to say "I am not sure", instead of guessing.

## Results

Two evaluation sets were written and SHA-256 frozen **before** any training example existed
(`data/frozen.json`, 2026-11-22) and were scored exactly once, on the shipped adapter.

| model | in-distribution (30) | unseen-style OOD (30) |
|---|---|---|
| Qwen3 1.7B, identical prompt, no fine-tune | 12/30 (40.0%) | 7/30 (23.3%) |
| this adapter, fp16 | **29/30 (96.7%)** | **18/30 (60.0%)** |
| this adapter, Q4_K_M GGUF - the file you download | 28/30 (93.3%) | 14/30 (46.7%) |

+56.7 points in-distribution and +36.7 out-of-distribution in fp16; in the published Q4_K_M
file, +53.3 and +23.3. Quantisation costs one in-distribution example but four
out-of-distribution ones, so it erodes the harder generalisation first. Exact match means the whole record
agrees: the set of part/action/amount jobs, the cancelled set, and the clarification flag.

What improves is **not** formatting. The prompt-only model already returns valid JSON (30/30
in-distribution); it fails on *content*. The largest single change is the clarification flag,
which goes from 17/30 to 30/30 in-distribution and 13/30 to 28/30 out-of-distribution. No answer
on any split contains an invented garment part, so wrong answers are wrong choices from a closed
set rather than free-form hallucination.

On a 40-example development split (used only during development) the same adapter scores 40/40,
5/5 in each of the eight behaviour families. Earlier runs of this recipe scored 38/40, so the
honest band is 38-40; 40/40 is the top of it, not a generalisation claim.

On a separate post-development, web-informed 32-case stress test, the unmodified Qwen3 1.7B scored 7/32 (21.9%) versus 22/32 (68.8%) for our shipped Q4_K_M model. Fine-tuning corrected 15 previously wrong cases without regressing any previously correct scored cases. However, unsupported alteration concepts can be force-mapped, and explicit negation remains a safety-relevant failure.

## Before and after

A real held-out pair, both outputs verbatim:

```
Shalwar ka pait dhaai inch kam karein.

base  : {"jobs": [], "cancelled_parts": [], "needs_clarification": false}
tuned : {"jobs": [{"part": "waist", "action": "tighten", "amount_inches": 2.5}],
         "cancelled_parts": [], "needs_clarification": false}
```

The base model's most common failure is not malformed JSON - it is *silent* JSON. It returns a
valid record claiming there is nothing to do. Downstream that looks like success.

Seventeen of the 30 in-distribution sentences flip from wrong to right in this way. Every number
in this entry can be re-checked: the model repository carries `VERIFY.md` with the checksums,
the frozen evaluation hashes, the training configuration and the per-file scores.

## Dataset

345 examples, all authored for this task in Roman Urdu: 245 training, 40 development, 30
in-distribution evaluation, 30 out-of-distribution evaluation. No public dataset, no scraping,
no customer records.

- Each label is validated against the JSON schema in CI-style checks: **0 errors**.
- Splits were audited for near-duplicates across train/dev/eval: **none above 0.9 similarity**.
- The two evaluation sets were hashed before the training set was written, so they could not be
  tuned against.
- `data/VOCAB.md` is the labelling rulebook (number words, part synonyms, family rules) and is
  always extended *before* the examples that depend on it.

Small on purpose, and measured rather than asserted: moving from 120 to 220 to 245 examples
changed the score by **zero** each time. The ceiling for this task is pattern coverage, not
volume, so the last 25 examples targeted specific gaps (two edits in the same region, rare
spellings, negation plus edit) instead of adding bulk.

## Training

QLoRA, trained on a free Colab T4 because the local machine cannot hold the training step.

| | |
|---|---|
| base | `Qwen/Qwen3-1.7B`, 4-bit NF4, double quantisation |
| LoRA | r=16, alpha=32, all linear layers, 17.4M trainable parameters |
| step | fp16 compute (T4 is sm_75 with no native bf16 tensor cores), chunked logits |
| schedule | 3 epochs, lr 2e-4, effective batch 4, maxlen 1152 |
| time | about 15 minutes: 186 optimizer steps at the 4.9 s/step the 90-step pilot measured (445 s) |

Two details that cost real debugging and are easy to repeat:

- The model is always given its ~950-token system prompt. Without it the same adapter scores
  **0/40**: it spends the whole budget on visible reasoning about a task it never sees stated.
  Thinking must also be off - `--think=false` in this setup.
- The adapter is merged by applying `W += (B @ A) * alpha / r` in fp32 directly rather than with
  peft, because Colab's torchao 0.10.0 makes `PeftModel.from_pretrained` raise
  `only versions above 0.16.0 are supported`. The merge was verified against an attached-adapter
  forward pass: maximum absolute logit difference 6.1e-4.

## Run it locally

```bash
huggingface-cli download rehanabrar777/qwen3-1.7b-roman-urdu-tailor-workorder tailor-roman-urdu-Q4_K_M.gguf Modelfile --local-dir tailor
cd tailor && ollama create tailor -f Modelfile
ollama run tailor --think=false
```

The `Modelfile` carries the full system prompt, and `FROM` names the GGUF in the same
directory - keep the two files together, because `ollama create` resolves `FROM` relative to the
Modelfile and reports a misleading `Error: 400 Bad Request: invalid model name` otherwise.

Pass `--think=false`, or the model spends its whole budget on visible reasoning instead of
answering.

Measured on the target machine - RTX 3050 Laptop 6 GB, i5-13420H, 16 GB RAM, Ollama 0.40.14 -
from Ollama's own counters, not wall time:

| | |
|---|---|
| weight file | 1,107,408,736 bytes (1.11 GB), GGUF v3, 310 tensors, Q4_K_M |
| prompt eval | 3138 tok/s (995 prompt tokens) |
| generation | 81.6 tok/s (78 tokens) |

Local headroom matters: this model needs roughly 3 GB of free Windows commit. With about 1 GB
free the same GGUF fails with `cudaMalloc failed: out of memory`. That is an environment limit,
not a model defect, and it is stated here rather than left as a surprise.

## What we learned, including what still fails

Out-of-distribution accuracy is 60%, and the 12 failures are not random. They are the sentences
whose labels use vocabulary the training set deliberately does not demonstrate - `girban`
(collar), `kalai` (wrist), `shana` (shoulder), `chain` (zip) - plus one order asking for
different edits to two garments in a single sentence. The one in-distribution miss is
`Asteen dedh inch choti kar dein`.

Three failures worth naming, because they bound the claim:

1. `jb nikal do` (pocket, remove) is answered with `needs_clarification: true`. Asking is
   defensible for a two-word input, but it is still wrong against the label.
2. `daman aur lambai dono cancel, bazu dedh inch chhota` keeps the parts and the cancellations
   right and gets the amount wrong (3 instead of 1.5). Quantity decoding inside compound
   sentences is the remaining weak spot.
3. Adding 100 training examples (120 to 220) moved the score by zero; adding 25 targeted ones
   fixed one pattern and broke another. Coverage beats volume, and both are visible in
   `experiments/RUNS.md`.

What we would do next: extend the vocabulary to the OOD terms with the same frozen-set
discipline, and handle multi-garment orders as separate records rather than one object.

## Team

Rehan Abrar - [@Rehan-Abrar](https://github.com/Rehan-Abrar). Model, data, training and
evaluation by the author; built on Qwen3-1.7B, Ollama and llama.cpp.
