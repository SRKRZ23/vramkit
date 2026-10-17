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
| int4 per-token | 3.88× | 0.958 | 0.29 |
| int4 + range-coder (lossless on top) | up to ~12× | 0.958 | 0.29 |

**Headline: int8 KV = ~2× smaller at >0.999 fidelity** (essentially free). int4 trades quality on
outlier-heavy data → motivates outlier-aware int4 (a real open problem). The range-coder step proves
exploitable redundancy survives quantization.

## Status (honest)
- ✅ Range coder: pure-stdlib, bit-exact roundtrip verified on random + real quantized data.
- ✅ Weight codec: −22% vs raw-4bit, lossless, beats zstd/xz, runs in ~2s on a MacBook.
- ✅ KV-cache int8: ~2× at near-perfect fidelity. int4+coder: large ratio, quality TBD on a real model.
- ⏳ Next: validate KV quality as **perplexity on a real model** (needs a GPU — dgx-spark); outlier-aware
  int4; a GPU dequant kernel so the compressed form is used live, not just stored.

## Layout
- `vramkit/rangecoder.py` — the entropy coder (encoder + decoder + self-test).
- `bench/bench_weights.py` — weight-codec benchmark vs zstd/xz with entropy floors.
- `bench/bench_kvcache.py` — KV-cache quant + codec benchmark with fidelity.

MIT. Built on a MacBook; GPU validation pending.
