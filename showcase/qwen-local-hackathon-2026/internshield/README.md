# InternShield

**Detect fake internship ads targeting Pakistani university students — locally, with Qwen3.**

## What it does

Pakistani university students frequently encounter fraudulent internship advertisements that demand upfront fees, CNIC copies, or personal details. InternShield takes an internship ad as input and returns a structured JSON verdict: `GENUINE`, `SUSPICIOUS`, or `LIKELY_SCAM`, along with red flags, credibility signals, and verification steps tailored to the Pakistani job market (SECP registry, Rozee.pk, Mustakbil.com).

## Why we built it

Fake internship scams are widespread in Pakistan and disproportionately target students who are desperate for experience. Existing scam detectors are generic and miss Pakistan-specific patterns like Gmail-only contact, vague company names, and CNIC copy demands. A lightweight local model keeps data private — no internship ad text leaves the device.

No fine-tuning was needed: Qwen3-0.6B already has strong instruction-following and JSON output capabilities. A carefully crafted system prompt with Pakistani context is sufficient for this classification task.

## Run it locally

**Requirements:** Ollama installed, ~500 MB RAM free.

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

**Input:** "Internship at XYZ Pvt Ltd. Send CV + CNIC copy to xyz123@gmail.com. Rs 2000 registration fee required."

**Output:**
```json
{
  "verdict": "LIKELY_SCAM",
  "red_flags": ["upfront registration fee", "CNIC copy demand", "Gmail-only contact", "vague company name"],
  "green_signals": [],
  "verify_steps": ["Search XYZ Pvt Ltd on SECP company registry", "Check Rozee.pk or Mustakbil.com for official listing", "Never pay fees for internships"],
  "summary": "This ad shows multiple classic Pakistani internship scam patterns and should be avoided."
}
```

**Limitation:** The model may misclassify ads from small legitimate companies with informal contact practices.

## What we learned

- Qwen3-0.6B handles structured JSON output reliably with `temperature: 0.1`
- Pakistan-specific context in the system prompt (SECP, Rozee.pk, local scam patterns) significantly improves relevance
- CPU-only inference on a mobile i7 is usable for short prompts (~10–20 seconds per response)

## Team

| Name | GitHub |
|------|--------|
| Abdul | [@abdul15x](https://github.com/abdul15x) |
