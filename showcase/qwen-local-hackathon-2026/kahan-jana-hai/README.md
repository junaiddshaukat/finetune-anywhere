# Kahan Jana Hai — Karachi bus query parser (Qwen2.5-0.5B, LoRA)

## What it does

A Karachi rider types a bus query the way people actually write: `saddar se gulshan
chorangi kaise jaun`, `bhai mahtab cniema jaana hai, abhi first chowrangi mein khara
hoon`, or the same in Urdu script. The model returns only the two stops:


```json
{"from": "Saddar", "to": "Gulshan Chorangi"}
```

Those two canonical names go into the route engine of **Kahan Jana Hai**, an offline
installable PWA and Capacitor Android app for Karachi bus routes. Without the model,
the app can only be searched by exact stop name; with it, a full messy sentence works.

## Why we built it

Karachi has no transit API and no reliable stop-name spelling, so riders ask
questions in Roman Urdu, where `Nazimabad`, `Naazimabad` and `Nazim Abad` are the same
place. The app already holds 344 routes and 1,631 stop names, but it needs to know
*which two stops* a sentence is about. Interpreting free text is the one part a
25 MB rule engine cannot do well, and it is exactly what a small fine-tuned model
does cheaply — which is why fine-tuning was needed rather than more code.

Deliberately, **the model never routes**. It only extracts stop names; the TypeScript
engine finds the buses. A 0.5B model cannot be trusted to remember bus numbers, and
this design means it does not have to.

## Dataset

- **Source:** the app's own stop registry, `normalized-stops.json` — 1,631 usable
  stop names derived from a 2010 Karachi route listing plus OpenStreetMap names
  (OSM data is ODbL — credit OpenStreetMap contributors).
- **Size:** 5,641 training rows, 201 held-out test rows, chat format (system / user /
  assistant). 575 rows teach the null answer (10.2%); one-stop rows are balanced
  (14.1% destination-only, 14.2% start-only).
- **How the queries were made:** generated from 30+ Roman Urdu, English and
  Urdu-script templates, then varied with Roman Urdu spelling variants
  (`se/say/sy`, `kaise/kese`, `mujhe/mjhe`, `bus/bas`), letter typos (never inside a
  digit), filler (`bhai`, `yaar`, `plz`), all-lower/Title/ALL-CAPS, partial queries
  that must return `null` on one side, and no-place queries that must return both
  nulls. 24 further phrasings were written by hand.
- **Review:** the 16 hand-written test queries were written by hand and never
  trained on. Stop labels are checked against the registry by script.
- **Split:** the test set uses **different templates and different stop pairs** from
  training and its own fixed seed, so it measures unseen wording rather than
  memorised templates, and it does not change when the training data is rebuilt.
- **License:** stop names come from app data we own plus OSM (ODbL). Queries are
  synthetic. No third-party dataset was used.

## Training

| | |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Tool | `transformers` 5.24.1 SFT path via `trl` 1.19.0 `SFTTrainer`, `peft` 0.22.0 |
| Platform | local laptop, no cloud GPU |
| Method | LoRA, r=16, alpha=32, dropout 0.05, `all-linear` |
| Hyperparameters | 3 epochs, effective batch 16 (8 × 2 accumulation), lr 2e-4 cosine, 30 warmup steps, max length 256, bf16 |
| Hardware | ASUS laptop, RTX 4060 Laptop GPU 8 GB, Ryzen 7 8845HS, 16 GB RAM |
| Measured duration | **27 min** (1,059 optimizer steps, final train loss 0.720) |

Two earlier attempts are worth reporting because they changed the outcome: effective
batch 32 trained in 17 min but reached only 423 steps and scored 50% — same data,
fewer updates, so the step count mattered more than the data changes. And an early
run taught only 131 unique no-place examples (2.9%), which left the model answering
invented stops to questions like "bus ka kiraya kitna hai".

## Run it locally

Runs on the laptop's RTX 4060, 100% GPU, no cloud. The shipped artifact is a
**Q4_K_M GGUF of 380 MB** (373.71 MiB, 6.3455 BPW, `general.file_type = 15`).

```bash
huggingface-cli download abdul1977ghaffar/karachi-bus-query-parser \
  karachi-q4_k_m.gguf Modelfile --local-dir kahan
cd kahan && ollama create kahan-jana-hai -f Modelfile
ollama run kahan-jana-hai "saddar se gulshan chorangi kaise jaun"
```

```
{"from": "Saddar", "to": "Gulshan Chorangi"}
```

Measured with Ollama 0.40.18: model loaded 100% on GPU (936 MB including a 32k
context), roughly 50–100 tok/s generation. Two queries that show the behaviour:

```
bus ka kiraya kitna hai                              -> {"from": null, "to": null}
tower jana hai                                        -> {"from": null, "to": "Tower"}
bhai mahtab cniema jaana hai, abhi first chowrangi..  -> {"from": "First Chowrangi", "to": "Mahtab Cinema"}
```

Reproduce the export (Windows, no C++ toolchain — prebuilt llama.cpp binaries):

```bash
python finetune/merge.py
python finetune/llama-src/convert_hf_to_gguf.py finetune/merged \
  --outfile finetune/karachi-f16.gguf --outtype f16
./finetune/llama-bin/llama-quantize.exe finetune/karachi-f16.gguf finetune/karachi-q4_k_m.gguf Q4_K_M
```

Note: `ollama create --quantize q4_K_M` does not work on this Ollama version (it
accepts only `int4/int8/nvfp4/mxfp4/mxfp8`), and importing safetensors directly
fails on Windows with `unsupported MLX architecture`. The GGUF must be produced by
llama.cpp first.

Reproduce the fine-tune from this repo:

```bash
python finetune/make_data.py     # build train/test jsonl from the stop registry
python finetune/eval.py base     # baseline
python finetune/train.py         # ~22 min on an 8 GB RTX 4060
python finetune/eval.py tuned    # after
python finetune/report.py        # writes results/RESULTS.md
```

## Before and after

Same held-out query the model never saw in training:

```
prompt:  bhai mahtab cniema jaana hai, abhi first chowrangi mein khara hoon
base:    {"start_stop": "Chowrangi", "destination_stop": "Bhuj"}
tuned:   {"from": "First Chowrangi", "to": "Mahtab Cinema"}
```

The base model picks the wrong keys and treats "mahtab cniema" as noise; the tuned
model returns both real stops, including the misspelled one.

Full held-out results (201 queries, identical for both runs):

| | Base | Fine-tuned |
| --- | ---: | ---: |
| Valid JSON | 100% | 100% |
| Both stops exactly right | **8%** | **65%** |
| Start stop right | 23% | 82% |
| Destination right | 18% | 78% |
| Case/punctuation-insensitive | 8% | 75% |
| Hand-written queries (16) | 12% | **81%** |
| Queries with no place (15) | 100% | **100%** |
| Destination-only (30) | 0% | 73% |
| Start-only (20) | 0% | 65% |

Judged by exact string match against the canonical stop name, which is stricter than
the app needs. Most remaining misses are 1–2 character slips (`Kharadar` →
`Kharkaradar`) that the app's typo-tolerant matcher repairs.

**Failure we found and fixed, stated with numbers:** an earlier run answered
invented stops to queries containing no place at all (`bus ka kiraya kitna hai`),
scoring 6/15 against the base model's 15/15. The cause was data, not the model: only
131 unique rows (2.9%) taught a null answer. Expanding that to 575 unique rows
(11.6%) took the group to **15/15**.

**What is still weakest:** one-stop queries still hinge on the wording, not the
place name. A mid-project regression showed this clearly: with too few one-stop
examples the model filed a lone stop under `from` even in
`need a bus for bahadurabad`, taking the destination-only group to 50%. Expanding
the one-stop templates and rebalancing the rows restored it to 73%, and the same
three failing queries now answer correctly in the shipped GGUF. The lesson is that
"which field does a single stop belong to" is a behaviour that needs explicit
surface area in the data.

## What we learned

- **Optimizer steps beat data tweaks.** Widening the data but doubling the effective
  batch cost 10 points; halving the batch recovered it and gained 3 more.
- **Casing is a real failure mode at 0.5B.** The model title-cases names
  (`Haji Camp` for `Haji camp`); training on mixed casing closed most of that gap —
  69% is the case-insensitive ceiling.
- **Do not ask a small model for the stop it "means".** Letting it echo the user's
  spelling and fuzzy-matching afterwards would have raised the score, but returning
  the canonical name directly is what the engine needs.
- **Stop registries are dirty.** `Abidabad Khadda Market` and `Khadda Market` are
  different rows, and the model confuses them; that is a data-cleanup problem, not a
  training one.
- **Nulls need explicit mass.** "Sometimes there is no answer" is a behaviour, not a
  gap the model infers: at 2.9% of rows it never learned it, at 11.6% it is perfect.
- Untested: GGUF quantization quality, on-phone inference, and Urdu-script accuracy
  beyond the 269 stops that have an alias.

## Links

- Model, adapter, GGUF and evaluation: <https://huggingface.co/abdul1977ghaffar/karachi-bus-query-parser>
- Training and evaluation data: the `dataset/` folder of that repo (`train.jsonl`, `test.jsonl`)
- App: Kahan Jana Hai (Karachi bus route PWA + Android). The app repository is private;
  the model repository above carries the adapter, Modelfile and per-group results.

## Team

- Muhammad Waqar — [@Muhammad-waqar-uit](https://github.com/Muhammad-waqar-uit)

Built with Qwen2.5, Hugging Face `transformers`, `trl`, `peft` and PyTorch. Stop
names include OpenStreetMap data (© OpenStreetMap contributors, ODbL). App:
Kahan Jana Hai.
