# PK-RestoQwen3-4B

A locally runnable Pakistani restaurant customer-support assistant, fine-tuned to understand English, Roman Urdu, Urdu script, and natural Urdu-English code-switching.

**Team:** Soman Abbasi and Ali Haider

**Model:** [AliHaider1/PK-RestoQwen3-4B](https://huggingface.co/AliHaider1/PK-RestoQwen3-4B)

## What it does

PK-RestoQwen3-4B is designed for Pakistani restaurant customer support. It handles common customer conversations, including delayed deliveries, duplicate orders, cancellation requests, complaints, and general restaurant questions.

The goal is to make local deployment practical for restaurants that want to reduce dependence on cloud-hosted language models and support the way customers actually communicate in Pakistan.

It is intended to understand:
- English
- Roman Urdu
- Urdu script
- Mixed Urdu-English conversations

## Why we built it

Many Pakistani customers naturally switch between Urdu and English, particularly when communicating with businesses through chat or messaging applications.

Generic assistants may respond in an unnatural language or fail to maintain the customer's conversational style.

We fine-tuned a compact Qwen model for this domain and language context, with local inference as the deployment target.

## Dataset

The training dataset contains approximately 1,500 curated restaurant customer-support examples.

It combines restaurant-domain conversational examples with localized Pakistani examples covering English, Roman Urdu, Urdu script, and Urdu-English code-switching.

The intended coverage includes order delays, duplicate orders, cancellation requests, customer complaints, and general restaurant support.

The dataset was curated and filtered for the task. It is not presented as a publicly released standalone dataset.

## Training

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Training framework: Hugging Face Transformers and PEFT
- Method: LoRA / QLoRA
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Epochs: 1
- Learning rate: `5e-5`
- Scheduler: cosine
- Weight decay: `0.01`
- Effective batch size: 16
- Loss: completion-only training
- Training platform: Google Colab
- Training GPU: NVIDIA Tesla T4

The published Hugging Face repository contains the fine-tuned adapter. The original base model is required alongside it.

## Run it locally

The model was tested locally using `llama.cpp` with a Q4_K_M GGUF base model and a separately converted GGUF LoRA adapter.

After downloading the base GGUF and adapter GGUF into a local `models/` directory, run:

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --lora ./models/PK-RestoQwen3-4B-LoRA.gguf \
  -c 1024 \
  -t 4 \
  --host 127.0.0.1 \
  --port 8080
```

The server exposes an OpenAI-compatible chat completion endpoint:

`http://127.0.0.1:8080/v1/chat/completions`

Tested on a local Intel Core i7-1355U CPU laptop with 8 GB RAM. In observed short tests, generation speed was approximately 4–6 tokens per second. Actual speed depends on prompt length, context, and system load.

## Before and after

A matched base-versus-fine-tuned comparison is being prepared using identical prompts and generation settings.

The comparison will include:
- Pakistani language understanding
- Urdu-English code-switching
- Restaurant customer-support intent handling
- Avoidance of fabricated order status or actions

We have observed that the fine-tuned model can understand common Roman Urdu and Urdu-script restaurant requests. Early tests also revealed limitations in natural Urdu phrasing, language matching, and claims about actions it cannot actually perform.

The model does not have live access to restaurant order systems. It must not claim to check, cancel, refund, or forward an order unless an actual integration confirms that action.

## What we learned

- A 4B model can be run locally on a CPU with a suitable GGUF quantization.
- Pakistani conversational localization needs explicit examples of Roman Urdu, Urdu script, and code-switching.
- Fine-tuning does not automatically provide access to business systems or live order information.
- Qualitative smoke tests are useful for finding issues, but a matched evaluation against the base model is necessary before claiming measurable improvement.

## Limitations

This is a prototype, not a production restaurant order-management system. It does not independently retrieve live menus, track deliveries, process cancellations, or issue refunds.

Its conversational quality and factual reliability need broader evaluation before deployment.

## Team

- **Soman Abbasi:** https://github.com/SomanAbbasi
- **Ali Haider:** https://github.com/AliHaiderno2

## Model and base model

- Fine-tuned model: https://huggingface.co/AliHaider1/PK-RestoQwen3-4B
- Base model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
