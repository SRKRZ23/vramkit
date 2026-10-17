# Launch post — vramkit

*(draft for HN / r/LocalLLaMA / X — tighten per venue)*

---

**Show HN: vramkit — lossless-verified VRAM compression for LLM inference, built on a MacBook**

At inference scale, VRAM is the binding cost: it decides how many GPUs you need and how long a
context you can hold. vramkit attacks the two things that eat it — **model weights** and the
**KV-cache** — with small, measured, bit-exact-verified techniques. Every number is produced by a
runnable benchmark with a roundtrip assertion; nothing is hand-waved.

**Weights — lossless, beats general compressors.** A from-scratch 32-bit range coder with a 2-D
adaptive context (each 4-bit weight conditioned on its left + up neighbour — structure zstd/xz can't
see): **3.12 bits/weight, −22% vs raw 4-bit, lossless**, beating zstd-19 and xz-9e, sitting on the
2-D entropy floor.

**KV-cache — quantize + entropy-code.** The KV-cache grows linearly with context and dominates memory
on long sequences. Per-token **int8 = ~2× smaller at cos-sim 0.9998** (essentially free). Naive int4
loses quality on KV's outlier channels; keeping just 4 of 128 channels in int8 — **outlier-aware
int4** — recovers cos-sim 0.958 → **0.999** at ~int4 memory (≈3.6×). A range coder on the int4 symbols
adds further lossless shrink.

The whole thing is pure-stdlib + numpy and was **built and benchmarked entirely on an Apple-Silicon
MacBook** — no datacenter GPU needed to reproduce it.

It's early and honest about limits: KV quality is reported as a cos-sim proxy (real-model perplexity
via MLX is the next step), and the live win needs a GPU dequant kernel so the compressed form is used
during attention, not just stored. Feedback and PRs welcome.

Repo: https://github.com/SRKRZ23/vramkit

---

**Short (X/Twitter):**
Built vramkit on my MacBook: lossless weight compression (−22% vs 4-bit, beats zstd) + KV-cache
compression (int8 ~2× near-lossless, outlier-aware int4 at cos-sim 0.999). VRAM is the cost of LLM
inference — every % is GPUs saved. All numbers bit-exact verified. https://github.com/SRKRZ23/vramkit
