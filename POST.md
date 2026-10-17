# Launch post — vramkit

*(draft for HN / r/LocalLLaMA / X — tighten per venue)*

---

**Show HN: vramkit — VRAM compression for LLM inference, measured as real perplexity on a MacBook**

At inference scale, VRAM is the binding cost: it decides how many GPUs you need and how long a
context you can hold. vramkit attacks the two things that eat it — **model weights** and the
**KV-cache** — with small techniques, each behind a runnable benchmark. Weights are bit-exact
lossless; KV is validated as **real end-task perplexity on a real model**, not a proxy.

**Weights — lossless, beats general compressors.** A from-scratch 32-bit range coder with a 2-D
adaptive context (each 4-bit weight conditioned on its left + up neighbour — structure zstd/xz can't
see): on real Qwen2.5-0.5B weights, **2.68 bits/weight, −33% vs raw 4-bit, lossless**, beating
zstd-19 by 1.5% and xz-9e by 1.2%, sitting on the 2-D entropy floor. (−22% on synthetic data; the
real-weights number is the one that counts.)

**KV-cache — the whole game is the key axis.** Keys carry persistent per-channel outliers, so a
per-**token** quant scale gets wrecked by them; keys want a per-**channel** scale, values are fine
per token. Measured as real perplexity on Qwen2.5-0.5B (KV compressed on every layer, 4 windows of
real prose):

- int8 per-token K&V (the naive way): **+4.6%** perplexity at 2×
- int8 **per-channel K** + per-token V: **+0.1%** at 2× — near-lossless
- int8 K + int4 V: **+1.4%** at 2.67×
- int4 keys: still +25% — int4 is a values-only lever

Getting the key axis right is the difference between +4.6% and +0.1% at the *same* memory.

**Honest correction baked in:** an earlier cut reported KV quality as cosine similarity and as a
perplexity number from a *repetitive* corpus, which made int8 look free and hid the per-token key
problem. cos-sim overstated quality and a repetitive corpus makes perplexity meaningless — both
fixed. The numbers above are diverse prose, end-task perplexity, multiple windows.

The whole thing is pure-stdlib + numpy + plain PyTorch and was **built and benchmarked entirely on
an Apple-Silicon MacBook** — no datacenter GPU needed to reproduce it. Open next steps: a grouped
per-channel key path for streaming decode (KIVI-style) and a Metal/ROCm dequant kernel so the
compressed form is used *during* attention. Feedback and PRs welcome.

Repo: https://github.com/SRKRZ23/vramkit

---

**Short (X/Twitter):**
Built vramkit on my MacBook. Finding: LLM KV-cache keys must be quantized PER-CHANNEL, not per-token
— their outlier channels wreck a per-token scale. Measured on a real model: per-channel int8 keys =
2× smaller KV at +0.1% perplexity; the naive per-token way costs +4.6%. Plus lossless weight
compression −33% vs 4-bit (beats zstd). All reproducible. https://github.com/SRKRZ23/vramkit
