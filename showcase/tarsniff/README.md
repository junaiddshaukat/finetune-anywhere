# tarsniff

Checks an npm package for malware **before you install it**, using a fine-tuned Qwen3-1.7B that
runs on your own laptop.

- Code: https://github.com/mohibovais79/tarsniff
- Model: https://huggingface.co/mohibovais79/tarsniff (GGUF Q4_K_M, 1.0 GB, plus the LoRA adapter)
- Slides: https://github.com/mohibovais79/tarsniff/blob/main/docs/tarsniff-slides.pdf

## What it does

A developer runs `tarsniff scan <package>` before `npm install`. tarsniff downloads the package
tarball (it never installs, extracts or runs it), builds a short digest of the code that matters
(install scripts first, then entry files, then code that touches the network, shell or secrets)
and asks the fine-tuned Qwen how likely the package is to be malicious. It prints the verdict, the
install scripts, the files it read and the risky lines (URLs redacted). It exits with code 1 when
a package is flagged, so `tarsniff scan X && npm install X` never installs a flagged package.

## Why we built it

`npm install` runs a package's install scripts automatically, so one malicious package can steal
tokens, SSH keys, cloud keys and crypto wallets. OSV published more than 1,400 malicious-npm reports
between 13 and 27 Nov 2026 alone. Rule-based scanners like GuardDog miss new code that matches no
rule, and the strongest LLM-based detectors in research call large cloud models.

We wanted a detector that reads the code like an analyst but runs offline on a normal laptop. A
small model needed fine-tuning: the base Qwen3-1.7B called 93% of safe packages malicious.

## Dataset

- **Malicious:** npm samples from the [Datadog malicious-software-packages dataset](https://github.com/DataDog/malicious-software-packages-dataset)
  (Apache-2.0, used with attribution), one version per package name.
- **Benign:** popular npm packages (GuardDog's top-npm list plus hand-picked "tricky" legitimate
  packages), downloaded from the registry.
- **Split by date** so one attack campaign can't be in both train and test: train = Jan–May 2026
  (300 malicious + 300 benign = 600 examples), validation = Jun–Oct 2026 (150 + 150).
- **Live test:** every npm malware report OSV published 13–27 Nov 2026 whose malicious version
  was still downloadable on 27 Nov (98 reports, 24 removed by npm before download, 74 scored),
  plus the latest versions of 286 popular packages as the safe control.
- **Review:** duplicates removed; package name, version and description stripped from every
  prompt so the model judges code, not names. All malware was handled only inside an offline,
  read-only Docker container; nothing was ever `npm install`-ed.

## Training

| | |
|---|---|
| Tool / platform | MLX-LM 0.32.2 on Apple silicon |
| Base | `mlx-community/Qwen3-1.7B-4bit` (4-bit MLX build of Qwen/Qwen3-1.7B) |
| Method | LoRA, rank 16, scale 20, dropout 0, all layers; loss on the answer only (`mask_prompt`) |
| Hyperparameters | 600 steps, batch 1, learning rate 1e-4, Adam, max length 1,200 tokens, gradient checkpointing, seed 0 |
| Input / target | 1,024-token package digest → `{"verdict": "malicious"}` or `{"verdict": "benign"}` |
| Hardware / time | Mac mini M4, 16 GB: 36 minutes, 2.9 GB peak memory |
| Export | `mlx_lm.fuse --dequantize` → llama.cpp `convert_hf_to_gguf.py` (bf16) → `llama-quantize Q4_K_M` |

We trained one LoRA per input size (file list only, 1,024, 2,048, 4,096 tokens). 1,024 tokens was
best on validation (F1 0.750 vs 0.641 / 0.683 / 0.709) and is the one shipped.

## Run it locally

Runtime: Ollama (tested 0.40.18) · quantization: Q4_K_M · model file 1.0 GB · context 4,096 ·
tested on Mac mini M4, 16 GB, about 1 s per package.

```bash
ollama pull hf.co/mohibovais79/tarsniff:Q4_K_M
pip install git+https://github.com/mohibovais79/tarsniff     # Python 3.10+
tarsniff scan lodash
tarsniff scan some-package@1.2.3 && npm install some-package@1.2.3
```

System prompt: *"You are an npm supply-chain security analyst. Given the contents of an npm
package, decide if it is malicious. Respond with JSON only."* The CLI builds the digest and reads
P(malicious) from the first answer token; threshold 0.5.

## Before and after

Same prompts, held-out packages, base Qwen3-1.7B (4-bit) vs fine-tuned.

**Live npm malware (Nov 2026) + popular safe packages** (packages with code: 73 malicious, 280 safe):

| | Base | Fine-tuned |
|---|---|---|
| Safe packages wrongly flagged | 93% | **9%** |
| Malware caught | 96% | 85% |
| F1 | 0.35 | **0.77** |
| ROC-AUC | 0.82 | **0.95** |

**Held-out validation (Jun–Oct 2026, 300 packages):** base F1 0.673 with a 96% false-alarm rate →
fine-tuned F1 0.750 with 6.7%.

**Example, safe package `@csstools/css-tokenizer@4.0.3`:**
- Base: `{"isMalicious": true, "reason": "The package contains a class ParseError with methods that could be used for error injection…"}`
- Fine-tuned: `{"verdict": "benign"}`

**Against GuardDog 3.2.0 on the live set:** tarsniff caught 62/74 attacks (GuardDog 45/74; both
together 70/74), but flagged 29/286 popular safe packages (GuardDog 5/286).

**Failures:** it missed a hijacked release of the real `vant` UI library and other large packages
where the malicious code sits beyond the 1,024 tokens it reads. On the NPMStudy benchmark sample,
its false-alarm rate at threshold 0.5 was 27% (F1 0.862 vs GuardDog 0.922).

We judged outputs by the score on the first answer token against the known labels, with the
threshold fixed at 0.5 before looking at the live results.

## What we learned

- Fine-tuning mattered most for false alarms: the base model flags nearly everything.
- More code did not help a 1.7B model: 1,024 well-chosen tokens beat 2,048 and 4,096.
- Fine-tuning on one benchmark's distribution overfit: a variant trained on NPMStudy data scored
  higher there but worse on newer 2026 malware, so we shipped the date-split model.
- Rules and the model catch different attacks; running both is best.
- Still open: catching malicious code hidden deep inside large, popular packages.

## Team

Mohib Ahmed ([@mohibovais79](https://github.com/mohibovais79)).

Credits: Qwen team (Qwen3-1.7B), Datadog Security Labs (malicious-software-packages dataset,
GuardDog), OSV / OpenSSF malicious-packages, MLX-LM, llama.cpp, Ollama.
