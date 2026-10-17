# vramkit — measurable VRAM compression for LLM inference

Two complementary, **measured, lossless-verified** techniques to cut the two things that eat GPU
memory during LLM inference: the **model weights** and the **KV-cache**. Every number below is
produced by a runnable benchmark with a bit-exact roundtrip assertion — no hand-waving.

Why it matters: at inference scale, VRAM is the binding cost. Shrinking weights and KV-cache
directly cuts GPU count / lets bigger context fit — the exact lever hyperscalers spend heavily on.

## Results (reproduce: `python3 bench/bench_weights.py`, `python3 bench/bench_kvcache.py`)

### 1. Weight codec — lossless, beats general compressors
A from-scratch 32-bit range coder with a 2-D adaptive context model (each 4-bit weight conditioned
on its left + up neighbours) — structure byte-oriented compressors (zstd/xz) can't see.

| method | bits/weight | vs raw 4-bit |
|---|---|---|
| raw 4-bit packed | 4.000 | — |
| zstd -19 | 3.139 | −21.5% |
| xz -9e | 3.162 | −20.9% |
| **vramkit (2-D ctx)** | **3.121** | **−22.0%** |

Lossless (bit-exact verified). Beats zstd-19 and xz-9e; sits on the 2-D entropy floor.
(Prior real-model validation: ~1 pt lossless edge over zstd-19 on 4-bit Qwen2.5-Math-7B, on a MacBook.)

### 2. KV-cache compression — quantize + entropy-code
Per-token int8/int4 quantization (handles KV's outlier-channel structure), optionally range-coded.

| method | vs fp16 | cos-sim | rel-err |
|---|---|---|---|
| fp16 (baseline) | 1.00× | 1.000000 | 0.0 |
| **int8 per-token** | **1.97×** | **0.9998** | 0.019 |
| int4 per-token (naive) | 3.88× | 0.958 | 0.29 |
| **int4 + int8 outliers (4 ch)** | **3.66×** | **0.9989** | 0.046 |

**Headline:** int8 KV = ~2× at >0.999 fidelity (essentially free). Naive int4 loses quality on KV's
**outlier channels**; keeping just 4 of 128 channels in int8 (**outlier-aware int4**) recovers cos-sim
0.958 → **0.999** at nearly int4 memory — the real engineering win. A range-coder on the int4 symbols
adds further lossless shrink (exploitable redundancy survives quantization).

## Status (honest)
- ✅ Range coder: pure-stdlib, bit-exact roundtrip verified on random + real quantized data.
- ✅ Weight codec: −22% vs raw-4bit, lossless, beats zstd/xz, runs in ~2s on a MacBook.
- ✅ KV-cache: int8 ~2× near-lossless; outlier-aware int4 3.66× at cos-sim 0.999 — all measured.
- Built and benchmarked **entirely on a MacBook (Apple Silicon)** — no datacenter GPU needed to
  reproduce the numbers.
- ⏳ Next (Mac-runnable): validate KV quality as **perplexity on a real small model via MLX**;
  push outlier-aware int4 lower; a Metal/ROCm dequant kernel so the compressed form is used live.

## Layout
- `vramkit/rangecoder.py` — the entropy coder (encoder + decoder + self-test).
- `bench/bench_weights.py` — weight-codec benchmark vs zstd/xz with entropy floors.
- `bench/bench_kvcache.py` — KV-cache quant + codec benchmark with fidelity.

MIT. Built on a MacBook; GPU validation pending.
