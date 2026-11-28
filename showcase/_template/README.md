# SafeShell-PowerShell Command Safety Assistant

## What it does

SafeShell is an experimental AI assistant that analyzes PowerShell commands and identifies potentially dangerous operations before a user runs them.

It produces a structured JSON assessment containing:
- Risk level: LOW, MEDIUM or HIGH
- Whether human review is required
- A short explanation of the risk
- A safer alternative when appropriate

SafeShell also uses rule-based command-chain detection and impact analysis to identify potentially destructive operations across multi-stage commands.

**Important:** SafeShell analyzes command text only. It does not execute, intercept or block terminal commands. It is a decision-support tool, not a security boundary.

## Why we built it

PowerShell is useful for system administration and automation, but commands involving recursive deletion, disk operations, permission changes, security settings and remote execution can cause serious damage when misunderstood.

We built SafeShell to make command risks easier to understand before execution.

Fine-tuning was used to adapt a small Qwen model to produce concise, structured command-risk assessments in a consistent JSON format rather than relying only on general-purpose language generation.

## Dataset

We created a synthetic dataset of 48 PowerShell command examples covering three risk categories:

- LOW: read-only or harmless operations
- MEDIUM: consequential but generally limited operations
- HIGH: destructive, broad-scope or security-sensitive operations

The dataset was split into:
- Training: 33 examples
- Validation: 6 examples
- Evaluation: 9 examples

The split was stratified using seed 42. We checked for duplicate prompts and overlap between splits.

The dataset was created for this project; it is not a publicly sourced benchmark. No external dataset license is claimed.

## Training

- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- Training platform: Google Colab
- Hardware: NVIDIA Tesla T4 GPU (14.56 GB VRAM)
- Framework: Unsloth with Transformers
- Method: 4-bit QLoRA / LoRA fine-tuning
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable parameters: 17,432,576 (approximately 1.003%)
- Epochs: 3
- Optimizer steps: 27
- Measured training duration: approximately 73.91 seconds
- Peak allocated GPU memory: approximately 4.24 GiB

Training loss decreased across the three epochs. Validation loss also decreased, from approximately 1.48 to 0.18.

The held-out evaluation achieved 7 correct classifications out of 9 (77.8%). All 9 outputs passed the JSON/schema and review-policy consistency checks. Both classification errors were MEDIUM examples overclassified as HIGH.

These results are preliminary because the evaluation set is small.

## Run it locally

### Download the model

The quantized GGUF model is available here:

[SafeShell on Hugging Face](https://huggingface.co/YFatima/safeshell-qwen3-1.7b-gguf)

Install the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Download the GGUF artifact:

```bash
hf download YFatima/safeshell-qwen3-1.7b-gguf safeshell-merged-16bit.Q4_K_M.gguf --local-dir models/safeshell
```

### Run with llama.cpp

Install or build llama.cpp and run:

```bash
llama-cli -m models/safeshell/safeshell-merged-16bit.Q4_K_M.gguf -p "Analyze this PowerShell command: Get-ChildItem C:\\Temp | Remove-Item -Recurse -Force. Return JSON with risk_level, requires_human_review, explanation, and safer_alternative. Use only LOW, MEDIUM, or HIGH for risk_level." -n 256
```

The exported GGUF uses Q4_K_M quantization and is approximately 1.03 GB.

We verified CPU inference using llama.cpp in Google Colab. CPU inference on the team's own laptop has not yet been independently benchmarked. Actual RAM requirements and performance will vary with the runtime and context length.

## Before and after

**Held-out command:**

```powershell
Get-ChildItem C:\Temp | Remove-Item -Recurse -Force
```

This command lists items in `C:\Temp` and pipes them to recursive, forced deletion.

**Base model:** The original model produced a lengthy natural-language response that did not complete the required JSON structure in the recorded comparison.

**Fine-tuned model:** The fine-tuned model returned a valid structured JSON assessment:

```json
{
  "risk_level": "HIGH",
  "requires_human_review": true,
  "explanation": "Permanently deletes files and folders in the specified directory.",
  "safer_alternative": "Get-Item C:\\Temp\\*"
}
```

The fine-tuned output correctly identified the destructive operation and followed the expected schema. The safer alternative is a read-only inspection command, not a direct replacement for the deletion operation.

We also evaluated nine held-out examples: seven classifications were correct, and two MEDIUM examples were overclassified as HIGH.

**Limitation:** The small evaluation set does not establish broad reliability. The model can misclassify commands, and complex PowerShell syntax may require additional parsing and testing.

## What we learned

- Fine-tuning improved the consistency of structured JSON output in our recorded comparison.
- A small, balanced dataset was enough to demonstrate the training workflow, but not enough to establish production-level accuracy.
- Rule-based command-chain detection adds useful visibility into multi-stage commands.
- Combining model assessments with deterministic checks can make the results more informative.
- MEDIUM-versus-HIGH classification remains an area for improvement.
- SafeShell should always be treated as an advisory tool with human review, not as a replacement for security controls.

## Team

**Yusra Fatima** — [GitHub: Usrafatima](https://github.com/Usrafatima)

Project model: [YFatima/safeshell-qwen3-1.7b-gguf](https://huggingface.co/YFatima/safeshell-qwen3-1.7b-gguf)

Tools and technologies: Qwen, Unsloth, Hugging Face, Google Colab, llama.cpp, Python and Gradio.

