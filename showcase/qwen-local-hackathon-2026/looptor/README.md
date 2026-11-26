# LoopTor — teaching Qwen3-1.7B to think without looping

## What it does

LoopTor is a LoRA fine-tune of Qwen3-1.7B (thinking mode) that fixes the model's
most common failure: getting stuck in thought loops — repeating the same
reasoning again and again until the token budget runs out and no answer is
ever delivered. LoopTor teaches three habits instead: think in proportion to
the question, verify once and then move on, and notice repetition, break out
of it, and conclude with an answer.

The fine-tuned model is exported to GGUF Q4_K_M and runs 100% locally in
Ollama, offline, on ordinary hardware.

## Why we built it

Repetition loops are the #1 practical complaint about small thinking models:
they burn tokens, waste minutes, and truncate without answering — worst in
coding help and long sessions, exactly where a local model should shine.

We measured it before changing anything. On LoopBench, our own benchmark of
unseen prompts, the base Qwen3-1.7B loops on:

- **56% of test prompts** (n=200)
- **85% of stress prompts** (n=100, loop-trap style questions)

A typical failure: the model reaches the correct answer inside its thinking,
then keeps second-guessing itself ("but maybe the user wants...") until it
hits the token cap and the answer is never delivered.

## Dataset

1,263 examples, all self-generated or programmatic — no real user data:

1. **Best-of-N rejection sampling (215):** for each training prompt the base
   model generated 6 samples; we kept the shortest correct, loop-free trace
   (SEER-style, arXiv 2509.14093).
2. **Programmatic concise traces (700):** short, verifiable reasoning for
   math, counting and logic.
3. **Loop-recovery examples (240):** the thinking starts to repeat, notices
   it, stops, and concludes.
4. **Direct-answer examples (108):** trivial questions answered with no
   thinking at all.

LoopBench is split into train pool (600) / unseen test (200) / stress (100)
with zero prompt overlap between splits. A response counts as a LOOP if any
8-gram repeats 3+ times, or any sentence of 10+ words repeats 3+ times, or
generation hits the token cap with no final answer (code blocks masked). The
detector was frozen before any results were measured.

## Training

- Base model: Qwen3-1.7B, thinking mode, 4-bit load
- Unsloth LoRA: r=16, alpha=32, all-linear, 3 epochs, lr 2e-4, max_len 1024,
  loss on responses only
- Platform: Kaggle free GPU (T4); Best-of-N generation started on Google Colab T4
- Export: GGUF Q4_K_M for Ollama

## Run it locally

```bash
ollama create looptor -f Modelfile   # points at the published GGUF
ollama run looptor
```

Model: https://huggingface.co/Daimx1214/looptor-1.7b

## Before and after

Base model (measured, LoopBench): test loop rate **56%**, stress loop rate
**85%**; accuracy 90.3% (test) / 28.3% (stress).

Fine-tuned model (measured, same LoopBench, same frozen detector): test loop
rate **41.5%**, stress loop rate **77%**; accuracy **93.6%** (test) /
**33.3%** (stress) — accuracy rose, so the accuracy guard passed. Local
20-prompt Ollama check agrees: loops 9/20 → 8/20, median tokens 1605 → 1328.

**Prompt (LoopBench test-017, unseen):** Ahmed is taller than Omar. Omar is taller than Fatima. Who is the shortest of the three? Explain the chain of comparisons carefully.

**Base Qwen3-1.7B (665 tokens, detector: LOOP):** [thinking] Okay, let's see. The problem says Ahmed is taller than Omar, and Omar is taller than Fatima. So, we need to figure out who is the shortest among the three... the order from tallest to shortest would be Ahmed, Omar, Fatima... [the same chain is then re-derived four more times in the thinking: transitive-relationship check, trick check, ambiguity check; the frozen loop detector flags repeated 8-grams] ...Therefore, the or[der...] (665 tokens; full output: project repo results/pair-test-017/base.json)

**LoopTor (583 tokens, detector: clean):** [thinking: one pass, no repeats] To determine who is the shortest among Ahmed, Omar, and Fatima, we start with the given comparisons:

1. **Ahmed > Omar**  
2. **Omar > Fatima**

These comparisons form a chain of inequalities:

$$
\text{Ahmed} > \text{Omar} > \text{Fatima}
$$

This chain clearly shows the order of their heights from tallest to shortest:

- **Tallest**: Ahmed  
- **Middle**: Omar  
- **Shortest**: Fatima

---

### Final Answer:

**Fatima** is the shortest of the three. (583 tokens; full output: project repo results/pair-test-017/tuned.json)

## What we learned

- Repetition loops are reduced, not eliminated: the biggest gains are on logic (loops 29/45 -> 14/45) and counting (47/48 -> 36/48); math barely moved (26/93 -> 24/93) and the hardest stress traps still loop (77%) - one LoRA pass on 1,263 examples teaches proportionate thinking, not new reasoning power.
- Non-termination fell sharply (test 10% -> 4.5%, stress 52% -> 39%): the model learned to end with an answer even when it still re-checks.
- Accuracy rose while tokens fell, so shorter did not mean sloppier - the Best-of-N shortest-correct traces were the highest-value data.
- Sampling variance is real: the same prompt can flip verdicts between runs, which is why the headline numbers come from 200/100-prompt evals, not single examples.

## Team

Daim Ali (solo) — https://github.com/Daimx1214
Project repo: https://github.com/Daimx1214/looptor
