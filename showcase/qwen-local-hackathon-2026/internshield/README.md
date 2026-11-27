# InternShield

**Fine-tuned Qwen3-0.6B to detect fake internship ads targeting Pakistani university students.**

Verdict accuracy improved from **9.1% → 57.1%** (+48.1 percentage points) after LoRA fine-tuning on a Pakistan-specific dataset.

## What it does

Pakistani university students frequently encounter fraudulent internship advertisements that demand upfront fees, CNIC copies, or personal details. InternShield takes an internship ad as input and returns a structured JSON verdict: `GENUINE`, `SUSPICIOUS`, or `LIKELY_SCAM`, along with red flags, credibility signals, and verification steps tailored to the Pakistani job market (SECP registry, Rozee.pk, Mustakbil.com).

## Why we built it

Fake internship scams are widespread in Pakistan and disproportionately target students who are desperate for experience. Existing scam detectors are generic and miss Pakistan-specific patterns like Gmail-only contact, vague company names, and CNIC copy demands. A lightweight local model keeps data private — no internship ad text leaves the device.

## Fine-tuning

The base `Qwen/Qwen3-0.6B` model was fine-tuned using **Unsloth LoRA** on Google Colab (Tesla T4 GPU).

**Training notebook:** https://colab.research.google.com/drive/1FhKXoDQUHI9i6iN722ZG-RkpfvIAKc_a?usp=sharing

**Dataset (953 examples total):**
- 719 rows from [agnialf/Internship-Job-Scam-Dataset](https://huggingface.co/datasets/agnialf/Internship-Job-Scam-Dataset)
- 199 rows from [victor/real-or-fake-fake-jobposting-prediction](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction) (internship-related)
- 35 handcrafted Pakistan-specific synthetic examples (upfront fees, CNIC demands, Gmail-only contact, government impersonation, Easypaisa scams)
- Split: 80% train / 10% validation / 10% eval

**Training config:**
- Method: LoRA (r=16, lora_alpha=16)
- Steps: 200, Epochs: 3
- Learning rate: 2e-4, Batch size: 2, Gradient accumulation: 4
- Final train loss: 1.6505

**Results:**

| Metric | Base Qwen3-0.6B | Fine-tuned InternShield |
|--------|----------------|------------------------|
| Verdict accuracy | 9.1% | **57.1%** |
| Improvement | — | **+48.1 pp** |

The fine-tuned model was exported as Q4_K_M GGUF using Unsloth's built-in export → `InternShield.Q4_K_M.gguf`.

## Run it locally

**Requirements:** Ollama installed, ~1 GB RAM free.

```bash
# 1. Place InternShield.Q4_K_M.gguf and Modelfile in the same folder
# 2. Create the model
ollama create InternShield -f Modelfile

# 3. Run it
ollama run InternShield
```

Paste any internship ad text as your prompt. The model returns a JSON object only.

- **Runtime:** Ollama
- **Quantization:** Q4_K_M
- **RAM required:** ~1 GB
- **Context length:** 512 tokens (output)
- **System prompt:** Baked into the Modelfile

## Example output

**Input:**
```
URGENT! Digital Marketing Hub Karachi. Salary Rs 60,000/month, no experience needed.
Send CV + Rs 2,000 registration fee to xyz@gmail.com
```

**Base model output (before fine-tuning):**
```json
{
  "verdict": "Genuine",
  "red_flags": [],
  "green_signals": [],
  "verify_steps": ["Check if the company is listed in the SECP registry"]
}
```

**InternShield output (after fine-tuning):**
```json
{
  "verdict": "LIKELY_SCAM",
  "red_flags": ["upfront registration fee", "Gmail-only contact", "unrealistic salary for no-experience role"],
  "green_signals": [],
  "verify_steps": [
    "Never pay registration fees — legitimate companies do not charge to hire",
    "Search the company on SECP registry (search.secp.gov.pk)",
    "Check on Rozee.pk for official listing"
  ],
  "summary": "This ad shows multiple classic Pakistani internship scam patterns — do not pay or share personal documents."
}
```

## What we learned

- Unsloth LoRA fine-tuning on a T4 GPU takes under 10 minutes for 200 steps on a 953-example dataset
- The base Qwen3-0.6B scored only 9.1% verdict accuracy — it needed domain-specific training, not just prompting
- Pakistan-specific synthetic examples were essential: without them the model missed local patterns like Easypaisa scam flows and CNIC copy demands
- Q4_K_M quantization gives ~500 MB file size with no noticeable quality loss vs full precision for this task
- CPU-only inference on a mobile i7 is usable for short prompts (~10–20 seconds per response)

## Team

| Name | GitHub |
|------|--------|
| Abdul | [@abdul15x](https://github.com/abdul15x) |
