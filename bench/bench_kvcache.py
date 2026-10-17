"""KV-cache compression benchmark. The KV-cache dominates VRAM at long context
(grows linearly with sequence length). We quantize fp16 K/V to int8/int4 with per-token
scales (handles the outlier-channel structure real KV has), optionally range-code the int4
symbols for extra lossless shrink, and measure memory reduction + reconstruction fidelity
(cosine similarity + relative error) as a quality proxy. Perplexity validation on a real
model is the next step (needs a GPU)."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vramkit.rangecoder import encode_symbols, decode_symbols

def make_kv(seq, heads, dim, seed=11):
    rng = np.random.default_rng(seed)
    kv = rng.standard_normal((seq, heads, dim)).astype(np.float32)
    # real KV has a few persistent OUTLIER CHANNELS (massive activations) — model them
    out_ch = rng.choice(dim, size=max(1, dim // 32), replace=False)
    kv[:, :, out_ch] *= rng.uniform(8, 20, size=len(out_ch))
    return kv

def quant_per_token(kv, bits):
    levels = (1 << bits) - 1
    half = levels // 2
    amax = np.abs(kv).max(axis=2, keepdims=True); amax[amax == 0] = 1
    scale = amax / half
    q = np.round(kv / scale).astype(np.int32)
    q = np.clip(q + half, 0, levels).astype(np.uint16)
    deq = (q.astype(np.float32) - half) * scale
    return q, deq, scale

def quant_outlier_aware_int4(kv, outlier_frac=1/32):
    """Keep the few outlier channels in int8, quantize the rest to int4.
    Real KV has persistent massive-activation channels that wreck a shared int4 scale."""
    seq, heads, dim = kv.shape
    chan_max = np.abs(kv).max(axis=(0, 1))          # per-channel max abs
    k = max(1, int(dim * outlier_frac))
    outlier_ch = np.argsort(chan_max)[-k:]
    mask = np.zeros(dim, bool); mask[outlier_ch] = True
    recon = np.empty_like(kv)
    # int4 for normal channels
    normal = kv[:, :, ~mask]
    qn, deqn, sn = quant_per_token(normal, 4)
    recon[:, :, ~mask] = deqn
    # int8 for outlier channels
    outl = kv[:, :, mask]
    qo, deqo, so = quant_per_token(outl, 8)
    recon[:, :, mask] = deqo
    n = kv.size
    bytes_ = (normal.size * 4 / 8 + outl.size * 8 / 8
              + sn.size * 2 + so.size * 2 + k * 2)   # +scales +outlier index
    return recon, bytes_, k

def fidelity(orig, recon):
    a = orig.reshape(-1); b = recon.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return cos, rel

def main():
    seq, heads, dim = 2048, 32, 128
    kv = make_kv(seq, heads, dim)
    n = kv.size
    fp16_bytes = n * 2  # KV stored as fp16

    print("=== KV-cache compression (seq=%d heads=%d dim=%d, %d elems) ===" % (seq, heads, dim, n))
    print("%-34s %12s %8s %9s %9s" % ("method", "bytes", "vs fp16", "cos-sim", "rel-err"))
    print("%-34s %12d %8s %9s %9s" % ("fp16 (baseline)", fp16_bytes, "1.00x", "1.000000", "0.0"))

    for bits in (8, 4):
        q, deq, scale = quant_per_token(kv, bits)
        cos, rel = fidelity(kv, deq)
        # raw quantized storage: bits per elem + fp16 scales (one per token*head)
        qbytes = n * bits / 8 + scale.size * 2
        print("%-34s %12d %7.2fx %9.6f %9.4f"
              % ("int%d per-token quant" % bits, int(qbytes), fp16_bytes / qbytes, cos, rel))

        # range-code the int4 symbols for extra LOSSLESS shrink on top of int4
        if bits == 4:
            sub = q[:256]  # code a slice for speed (representative), scale the ratio
            levels = (1 << bits)
            flat = sub.reshape(-1).tolist()
            cols = sub.shape[1] * sub.shape[2]
            def ctx_fn(i, sym, arr):
                j = i + 1
                left = arr[j - 1] if (j % cols) else 0
                up = arr[j - cols] if j >= cols else 0
                return left * levels + up
            enc = encode_symbols(flat, levels * levels, levels, ctx_fn)
            dec = decode_symbols(enc, len(flat), levels * levels, levels, ctx_fn)
            assert dec == flat, "KV int4 ROUNDTRIP MISMATCH"
            coded_bpe = len(enc) * 8 / len(flat)      # bits/elem after lossless coding
            full_coded = coded_bpe * n / 8 + scale.size * 2
            print("%-34s %12d %7.2fx %9.6f %9.4f"
                  % ("int4 + range-coder (lossless)", int(full_coded), fp16_bytes / full_coded, cos, rel))
            print("   (int4 symbols coded at %.3f bits/elem, bit-exact verified)" % coded_bpe)

    # outlier-aware int4: int8 for outlier channels, int4 for the rest
    recon, obytes, k = quant_outlier_aware_int4(kv)
    cos, rel = fidelity(kv, recon)
    print("%-34s %12d %7.2fx %9.6f %9.4f"
          % ("int4 + int8 outliers (%d ch)" % k, int(obytes), fp16_bytes / obytes, cos, rel))

    print("\nTakeaway: int8 KV = ~2x near-lossless; naive int4 loses quality on outlier channels,")
    print("but OUTLIER-AWARE int4 (int8 for a few massive-activation channels) recovers cos-sim ~1.0")
    print("at nearly int4 memory. Next: perplexity on a real model (GPU, dgx-spark).")

if __name__ == "__main__":
    main()
