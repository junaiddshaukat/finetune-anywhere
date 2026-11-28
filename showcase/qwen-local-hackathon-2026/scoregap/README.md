# ScoreGap

A fine-tuned **Qwen3-1.7B** (Q4_K_M GGUF, 1.1 GB) that reads a vulnerability description and drafts the **CVSS v3.1 base vector NIST's analysts would most likely assign, plus a CWE**. It runs offline on a laptop with llama.cpp.

[Model on Hugging Face](https://huggingface.co/faizanamir/scoregap-qwen3-1.7b-cvss) · [Code and app](https://github.com/faizanamer123/scoregap) · [Demo post](https://www.linkedin.com/posts/faizan-amir-88875a352_qwenlocalhackathon-ugcPost-7532399856154198017-bVNf/)

**In plain words.**
- **The problem.** Every new software vulnerability gets a public ID (a CVE), and teams decide what to fix first by its 0–10 severity score (CVSS). From mid-April to late November 2026 about 84,000 CVEs were published. NIST, which runs the U.S. vulnerability database, scored only about 1 in 10 of them.
- **What ScoreGap does.** Paste a description, and in under a second, offline on a laptop, it drafts the score NIST would most likely give. It explains each part and says how sure it is.
- **How well it works.** On 2,000 real 2026 vulnerabilities it never saw in training, it gets NIST's severity level right 70% of the time and within one level 96% of the time. Before fine-tuning, the same model managed 24%.

## What it does

A security analyst pastes an advisory or a CVE description, or uploads a CSV of them. ScoreGap returns:
- **A draft vector** such as `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`.
- **The score and severity**, computed by Python from that vector with the CVSS 3.1 equations, not written by the model.
- **A CWE**, with MITRE's name for it.
- **A plain-English meaning and a confidence for each metric**, so the reviewer knows which metrics to check first.

The model only chooses letters and a CWE id. A grammar limits it to valid vectors and real CWE ids. A Streamlit app adds CVE lookup in a local NVD snapshot, a base-vs-fine-tuned comparison and batch CSV export. Nothing leaves the machine.

**Who it is for.**

| Who | Use case |
| --- | --- |
| Security operations and vulnerability-management teams | Score a scanner's list of new CVEs, most of which now arrive without a NIST score. Upload a CSV, sort by severity and review the "check first" column; it takes about 13 minutes per 1,000 on a laptop. |
| Teams relying on the vendor's own score | Get a second opinion in NIST's style. On our test CVEs the vendor's severity was lower than NIST's 44.9% of the time. |
| Product security teams and CVE authorities | Draft the vector for an advisory before publication, offline, while details are still confidential. |
| Penetration testers and bug-bounty researchers | Draft a CVSS score for each finding without sending client details to a cloud service. |
| Banks, telecoms, government and air-gapped networks | Score vulnerabilities where cloud AI is not allowed. It even runs on a CPU alone, at 2.6 s per CVE. |
| Students and new analysts | Learn CVSS from per-metric plain-English explanations and compare them with NIST's real answers. |

**Example** (a made-up advisory, not from the test set): *"A stored XSS in the comment field of BlogPress 3.2 lets authenticated users inject script that runs in an admin's browser."*
- **Base Qwen3-1.7B:** `AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N`, 7.5 High, CWE-201. It assumes no login is needed and no victim has to act.
- **ScoreGap:** `AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N`, 5.4 Medium, CWE-79 (cross-site scripting). NIST gave this exact vector to 14 of the 19 similar CVEs in our 2026 test set.

## Why we built it

Vulnerability management tools rank work by the NVD's CVSS score, and NIST no longer produces one for most new CVEs:
- **Narrower scope.** Since 15 April 2026 NIST enriches mainly KEV, U.S.-government and EO 14028 "critical" software.
- **Few scores.** In our NVD snapshot only **9.6%** of CVEs published since then (8,081 of 84,417) have a NIST CVSS v3.1 vector.
- **The vendor's score often disagrees.** It is often the only score, and on our 923 test CVEs that both scored it matched NIST's severity band only **43.3%** of the time; it was lower on 44.9%.

A draft of NIST's likely score, made locally so private advisories never leave the machine, closes that gap for triage.

**Why fine-tuning was needed.** The base Qwen3-1.7B with the same instructions produced a valid vector only 63.7% of the time and matched NIST exactly on 2.4%. Even Qwen3-4B with worked examples reached only 19.1%. NIST's scoring conventions are learned from data, not from a prompt.

## Dataset

| | |
| --- | --- |
| Source | NVD JSON 2.0 yearly feeds 2019–2026, snapshot 2026-11-24, each file's SHA-256 checked against NIST's `.meta`. 292,695 non-rejected CVEs with English descriptions. |
| Label | The CVSS v3.1 entry with `source == "nvd@nist.gov"`; vendor and CISA scores are never used as labels. CWE: NIST's, else the vendor's, else `NVD-CWE-noinfo` (training only; evaluation uses NIST's CWE). |
| Text | One cleaning function for training, evaluation and the app. Embedded CVSS vectors are stripped (Oracle appends its own to every description) and text is cut at 2,000 characters. |
| Split | By publication date. **Test:** 2,000 CVEs published in 2026, drawn first and frozen with a recorded SHA-256. **Validation:** held-out 2025 CVEs. **Training pool:** 120,751 rows from 2019–2025. The shipped model trained on the **newest 80,000** (2022–2025); 79,966 remained after tokenisation. |
| Leakage control | Exact duplicates were removed, and any training text with TF-IDF cosine ≥ 0.90 to a test or validation text was purged when each pool was built. Re-measured against the shipped model's exact 80k rows (`scripts/test_similarity_r3a.py`), the median test CVE's closest training text has cosine 0.26. Four of 2,000, all vendor template advisories, reach 0.90–0.91; scoring them as wrong would move exact accuracy by 0.1 points. |
| Rights | NVD data are U.S. Government works. CVE and CWE text is © MITRE, used under its terms. CVSS is owned by FIRST. This product uses data from the NVD API but is not endorsed or certified by the NVD. |

## Training

| | |
| --- | --- |
| Tool, platform | Unsloth 2026.11.11 on a free **Kaggle T4** (Transformers 4.57.6, TRL 0.22.2, PEFT 0.21.2, PyTorch 2.11.0) |
| Base | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; thinking off in the chat template |
| Method | LoRA r 16, alpha 32, dropout 0, all attention and MLP projections, 16-bit base weights |
| Hyperparameters | 1 epoch = 9,996 steps, batch 8, lr 2e-4 cosine, 3% warm-up, AdamW 8-bit, max length 1,536 tokens, loss on the JSON answer only, seed 42 |
| Memory, loss | Peak VRAM 3.9 GiB; final eval loss 0.0501 |
| Export | Merged in bf16 → llama.cpp `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` (llama.cpp build 12359) |

**How the model was selected.**
- **Validation only.** Three fine-tunes were compared on 2025 validation data, using a rule written down before any run-3 result existed.
- **The 4B lost on the rule.** A Qwen3-4B QLoRA on the newest 30k rows tied the 1.7B (+0.1 points, 95% CI −1.4 to +1.6) at 2× the latency, so the 1.7B ships.
- **One test.** The test set was used once, after selection.

## Run it locally

Tested on a **MacBook Pro 14-inch, M1 Pro, 16 GB RAM, macOS 26.3.1**, llama.cpp Metal. Measured: 1.55 s to start, **0.76 s per CVE** in the app (median), 1.97 CVEs/s with 8 requests in parallel, 1.6 GB of RAM with one request slot (4.8 GB with 8), and 2.6 s per CVE on the CPU alone. System prompt: in `src/scoregap/prompts.py` and the model card. Greedy decoding, at most 96 new tokens. A 4,096-token context per request is ample, since prompts are at most about 2,500 characters.

```bash
git clone https://github.com/faizanamer123/scoregap.git && cd scoregap
python3 -m venv .venv && .venv/bin/pip install -e .
brew install llama.cpp
mkdir -p models/gguf
curl -L -o models/gguf/scoregap-1.7b-q4_k_m.gguf \
  https://huggingface.co/faizanamir/scoregap-qwen3-1.7b-cvss/resolve/main/scoregap-1.7b-q4_k_m.gguf
.venv/bin/scoregap "A SQL injection in the login form of Acme Portal 2.1 allows unauthenticated remote attackers to read the user database."
.venv/bin/scoregap --csv advisories.csv --out scored.csv      # batch
.venv/bin/streamlit run app.py                                # the app
```

## Before and after

**Test set.** 2,000 NIST-scored CVEs published in 2026, after all training data. Every language model gets the same system prompt and greedy decoding. ScoreGap's numbers come from the **Q4_K_M GGUF on the laptop**.

**How outputs were judged.** Automatically, against NIST's published vector. A system counts as right only on an exact match, and invalid output counts as wrong. Brackets are 95% bootstrap intervals.

How to read the table:
- **Exact vector:** all eight parts match NIST. One wrong letter is a miss, so this is the strictest measure.
- **Severity band:** the same None, Low, Medium, High or Critical level as NIST.
- **Mean per-metric:** the share of individual parts that are right.
- **CWE:** the weakness type, where NIST named one.

| System | Exact vector | Severity band | Mean per-metric | CWE (n=572) |
| --- | --- | --- | --- | --- |
| **ScoreGap** | **55.5%** [53.2–57.5] | **69.9%** | **89.6%** | **69.2%** |
| TF-IDF + logistic regression, **same 80k training rows** | 46.2% | 66.1% | 87.8% | 70.5% |
| Base Qwen3-1.7B, same prompt | 2.4% | 23.6% | 39.0% | 1.9% |
| Base Qwen3-1.7B + 8 retrieved examples | 27.0% | 49.2% | 77.6% | 39.7% |
| Base Qwen3-4B + 4 examples | 19.1% | 35.5% | 72.1% | 13.6% |
| Vendor's own vector (only the 923 CVEs where it exists; ScoreGap on those: 50.5% / 66.5%) | 6.7% | 43.3% | 77.9% | — |

- **Paired gain over the base model:** +53.1 points exact vector (95% CI +51.0 to +55.0).
- **Gain over a classical model trained on the same rows:** +9.2 (+7.5 to +11.2).
- **Gain over our previous fine-tune** (20k older rows): +6.0 points (+4.4 to +7.8).
- **Confidence is useful for triage.** On 2025 validation, drafts where every metric's confidence is ≥ 0.9 (a quarter of CVEs) are exactly right 90.7% of the time (expected calibration error 0.023).

Five test CVEs drawn at random, not chosen by hand (raw model text in the repo's `results/examples_tuned_r3a.json`):

| CVE | NIST | Base Qwen3-1.7B | ScoreGap |
| --- | --- | --- | --- |
| CVE-2026-49200: Acer firmware log with cleartext credentials, no auth | 9.8 CRITICAL | `C:H/I:N/A:N`, 7.5 (6/8) | **exact, 9.8 (8/8)** |
| CVE-2026-24502: Dell uncontrolled search path, local low-privilege | 7.8 HIGH | `AV:N/PR:N…`, 7.5 (4/8) | **exact, 7.8 (8/8)** |
| CVE-2026-70355: SharePoint XSS, privilege elevation | 8.7 HIGH | invalid (`PR:NORMAL/UI:NEED`) | 9.0 (7/8): `A:H` vs NIST `A:N` |
| CVE-2026-22690: pypdf long runtimes on malformed PDF | 5.3 MEDIUM | invalid (`PR:NORMAL`) | 7.5 (7/8): `A:H` vs NIST `A:L` |
| CVE-2026-59275: Spring AMQP, one hostile message ends the JVM | 4.9 MEDIUM | 7.5 (5/8) | 7.5 (7/8): `PR:N` vs NIST `PR:H` |

**Failures and limits.**
- **Typical errors.** The misses above are typical: impact overestimated on denial-of-service bugs, and privileges misread when the description does not say who can send the input. When ScoreGap's severity is wrong it is usually too high (20.9% of CVEs) rather than too low (9.2%).
- **Weak spots.** Short descriptions (under 150 characters) drop to 52.3% exact and 60.6% severity. CWE is less reliable than the vector, and there a TF-IDF classifier on the same rows is slightly better (70.5% vs 69.2%).
- **Biased test sample.** The 2026 test set covers what NIST still scores, so it is a biased sample of all CVEs.
- **A draft, not a verdict.** It predicts NIST, not the "true" risk.
- **Not a scanner, not a full priority list.** It scores vulnerabilities already described in text. It does not know whether a bug is being exploited or how important the affected system is, so pair it with CISA's KEV catalogue, EPSS and your own asset list.

## What we learned

- **Train on the population you test on.** Our first fine-tunes used only CVEs that also had a NIST CWE. That subset is shrinking: 27% of NIST-scored CVEs lacked a NIST CWE in 2023, 73% in 2026. The shipped run rebuilt the data around every NIST-scored CVE, took the **newest** years and four times as many rows, kept the base weights in 16-bit instead of 4-bit and raised the maximum length from 1,024 to 1,536 tokens. Together these gave +6 points on the frozen test; they were not ablated one by one. A TF-IDF proxy on validation had predicted recency as the strongest lever (+10.9 points for 2022–24 over 2019–21 at equal size).
- **A bigger model was not the answer.** Qwen3-4B fine-tuned on 30k rows tied the 1.7B on 80k rows. Prompting the 4B with examples stayed far behind any fine-tune.
- **Measure the quantised file, on the target machine.** Every number above is the Q4_K_M GGUF on the laptop.
- **Laptop operations matter.** Two operational lessons, both logged:
  - A recent llama.cpp default reserves a host prompt cache of up to 8 GB; we now start the server with `--cache-ram 0`.
  - Batch scoring on battery was more than 50× slower.
- **Uncertain.** We did not ablate data size against recency on the language model. A 4B on the same 80k rows might still win; we have not run it.

## Team

- [@faizanamer123](https://github.com/faizanamer123)
- [@abdullahiftikharcode](https://github.com/abdullahiftikharcode)

Credits:
- **Base model:** Qwen3-1.7B by the Qwen team (Apache-2.0).
- **Data:** NIST NVD; CVE and CWE by MITRE; CVSS by FIRST.
- **Tools:** Unsloth, Hugging Face Transformers, PEFT and TRL, llama.cpp, scikit-learn, Streamlit.
- **Platform:** Kaggle's free T4 GPUs.
- **Event:** the [finetune-anywhere](https://github.com/junaiddshaukat/finetune-anywhere) starter and organisers.
