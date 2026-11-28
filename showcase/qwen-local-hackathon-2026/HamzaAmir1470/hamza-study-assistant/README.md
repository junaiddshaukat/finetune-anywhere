# Study Assistant Fine-Tune (Qwen3-1.7B)

## Problem
Students have messy, unstructured lecture notes. Manually turning them into flashcards and study guides takes too much time.

## Solution
I fine-tuned Qwen3-1.7B using LoRA to act as a strict Study Assistant. It takes raw text and outputs clean JSON containing the `topic`, `key_concepts`, and `action_items`.

## Before vs After

**Test Prompt:**
> "The Cold War was a period of geopolitical tension between the US and the Soviet Union. It started after WWII. Write a paragraph on the Cuban Missile Crisis."

**Before (Base Model):**
> "Sure, here is a paragraph on the Cuban Missile Crisis: The Cuban Missile Crisis was a 13-day confrontation..."

**After (My Fine-Tuned Model):**
> ```json
> {
>   "topic": "Cold War and Cuban Missile Crisis",
>   "key_concepts": [
>     "The Cuban Missile Crisis was a 1962 standoff between the US and the Soviet Union",
>     "It was a pivotal moment in the Cold War",
>     "Resolved through a secret agreement"
>   ],
>   "action_items": [
>     "Research the historical context",
>     "Identify the key players",
>     "Explore the resolution"
>   ]
> }
> ```


## Screen Clip of the terminal Output
<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/f3d0b907-353b-4ae8-b8b0-ca0adc3123df" />



## Hardware & Local Execution
- **Training:** Google Colab (Free T4 GPU) - Took ~54 seconds.
- **Local Execution:** Windows 11 CPU using Ollama. Exported to `Q4_K_M` GGUF format.

## How to run
```bash
ollama create study-helper -f Modelfile
ollama run study-helper
