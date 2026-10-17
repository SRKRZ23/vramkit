# vramkit — measurable VRAM compression for LLM inference

Two techniques to cut the two things that eat GPU memory during LLM inference — the **model
weights** and the **KV-cache** — each with a **runnable benchmark** behind every number. Weight
numbers are **lossless, bit-exact verified**. KV numbers are validated as **real end-task
perplexity on a real model**, not a proxy. No hand-waving.

Why it matters: at inference scale, VRAM is the binding cost. Shrinking weights and KV-cache
directly cuts GPU count / lets bigger context fit — the exact lever hyperscalers spend heavily on.

Everything here was built and measured **entirely on a MacBook (Apple Silicon, CPU torch)** — you
do not need a datacenter GPU to reproduce it.

## 1. Weight codec — lossless, beats general compressors

A from-scratch 32-bit range coder with a 2-D adaptive context model (each 4-bit weight conditioned
on its left + up neighbours) — structure byte-oriented compressors (zstd/xz) can't see.

**On REAL LLM weights** (`Qwen2.5-0.5B`, `self_attn.o_proj`, int4 — `python3 bench/validate_real_weights.py <model.safetensors>`):

| method | bits/weight | vs raw 4-bit |
|---|---|---|
| raw 4-bit packed | 4.000 | — |
| zstd -19 | 2.717 | −32.1% |
| xz -9e | 2.708 | −32.3% |
| **vramkit (2-D ctx)** | **2.676** | **−33.1%** |

**Lossless (bit-exact verified), −33% vs raw 4-bit, beating zstd-19 by 1.5% / xz-9e by 1.2%** on
real quantized LLM weights — sitting on the 2-D entropy floor. Across **every** weight-tensor type
of Qwen2.5-0.5B it averages **2.92 bits/weight, −27% vs raw 4-bit**, lossless, beating zstd-19 on
4 of 5 — a general property of quantized LLM weights, not one lucky tensor. (On synthetic low-rank
data the edge over zstd is smaller, ~−22%; the real-weights number is the one that counts.)

## 2. KV-cache compression — get the quantization axis right

The headline finding, and the reason to read this repo: **keys and values want different
quantization axes.** Keys carry a few **persistent per-channel outliers** (massive-activation dims).
A per-**token** scale lets those outliers dominate and crushes every other value. A per-**channel**
scale (one per feature dim, across tokens) fixes it. Values are well-behaved per token.

**End-task validation — real perplexity, measured on a MacBook** (`python3 bench/validate_ppl_mac.py`,
Qwen2.5-0.5B, KV compression applied to **every** layer's K/V via forward hooks, averaged over 4
disjoint windows of real prose, baseline perplexity ≈ 18):

| scheme | memory | Δ perplexity vs fp32 KV |
|---|---|---|
| int8, per-token K & V (the naive way) | 2.0× | **+4.6%** |
| **int8, per-CHANNEL K + per-token V** | **2.0×** | **+0.1%  (near-lossless ✅)** |
| int8 K (per-chan) + int4 V (per-token) | 2.67× | +1.4% |
| int4, per-CHANNEL K + per-token V | 4.0× | +25.6% |

**The whole game is the key axis.** Per-channel int8 keys make the KV cache 2× smaller for **+0.1%
perplexity** — genuinely near-free. The naive per-token scheme most quick implementations reach for
costs **+4.6%** at the same memory. Pushing values to int4 buys 2.67× for ~1.4%; int4 **keys** still
regress hard (+25.6%), so int4 is a values-only lever here.

**Honesty note:** an earlier version of this repo reported KV quality as cosine similarity and as a
perplexity number from a *repetitive* eval corpus, which made int8 look free (+0.02%) and hid the
per-token key problem. cos-sim overstated quality (per-token vs per-channel keys differ by 0.0002 in
cos-sim but 4.5% in perplexity), and a repetitive corpus makes perplexity meaningless. Both are
fixed: the table above is real diverse prose, end-task perplexity, multiple windows. Reporting the
correction, not hiding it.

## Use it — drop-in KV compression (PyTorch, ROCm/CUDA/MPS/CPU)

`vramkit/kvquant.py` is a device-agnostic module — plain PyTorch, so it runs on **AMD/ROCm** (where
KV-cache is the long-context memory bottleneck), NVIDIA, Apple MPS, or CPU:

```python
from vramkit.kvquant import quantize_kv, dequantize_kv

pk = quantize_kv(key,   bits=8, per="channel")   # keys  -> per-channel (the important one)
pv = quantize_kv(value, bits=8, per="token")     # values-> per-token
key_hat   = dequantize_kv(pk)                     # reconstruct in the attention path
value_hat = dequantize_kv(pv)
```

`python3 vramkit/kvquant.py` runs the self-test (per-channel beats per-token on keys with outliers).
**Streaming decode:** per-channel key scales need the token axis, so quantize keys in groups of G
recent tokens (KIVI-style) with the newest few kept in fp16 — the numbers above are the quality
ceiling that grouping approximates.

## Why AMD / ROCm

On ROCm, long-context inference is a known pain point (the default attention path degrades hard past
~64k context, and FP8-KV support is still maturing). KV-cache memory is the lever: a near-lossless 2×
(or 2.67× with int4 values) means more concurrent requests or longer context per Instinct/Radeon GPU
— directly closing the gap with NVIDIA on $/token. vramkit is pure PyTorch + numpy, so it drops
straight into a ROCm stack.

## Status (honest)

- ✅ **Range coder**: pure-stdlib, bit-exact roundtrip verified on random + real quantized data.
- ✅ **Weight codec**: −33% vs raw-4bit on real Qwen weights (−22% synthetic), lossless, beats
  zstd-19/xz-9e, runs in ~2s on a MacBook.
- ✅ **KV-cache**: int8 **per-channel keys** = 2× at **+0.1% real perplexity** (near-lossless);
  int8-K/int4-V = 2.67× at +1.4% — all measured end-to-end on a real model.
- ✅ Built and benchmarked **entirely on a MacBook** — no datacenter GPU needed to reproduce.
- ⏳ Next: a grouped per-channel key path for streaming decode (KIVI-style), and a Metal/ROCm
  dequant kernel so the compressed form is used *live* in attention, not just stored. Seeking
  MI300X/Radeon time to validate on a 7B+ model and co-build the kernel.

## Layout
- `vramkit/kvquant.py` — drop-in KV compression (per-token / per-channel, int8 / int4) + self-test.
- `vramkit/rangecoder.py` — the entropy coder (encoder + decoder + self-test).
- `bench/validate_ppl_mac.py` — **real-model perplexity** of KV compression (the number that counts).
- `bench/validate_real_weights.py` — weight codec on real safetensors.
- `bench/bench_weights.py` / `bench/bench_kvcache.py` — synthetic benchmarks with entropy floors.

MIT. Built on a MacBook; GPU kernel + 7B validation are the open next steps.
