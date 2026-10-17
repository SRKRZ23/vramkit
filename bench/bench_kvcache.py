"""KV-cache compression benchmark (synthetic, cos-sim proxy). The KV-cache dominates VRAM
at long context (grows linearly with sequence length). Keys carry persistent per-channel
outliers, so a per-TOKEN scale wrecks them and a per-CHANNEL scale fixes them; values are
fine per token. We show that here on synthetic KV with injected outlier channels, and
range-code the int4 symbols for extra LOSSLESS shrink. cos-sim is only a proxy — the real
end-task perplexity numbers are in bench/validate_ppl_mac.py (real model, measured on a Mac)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vramkit.rangecoder import encode_symbols, decode_symbols


def make_key(seq, heads, dim, seed=11):
    """Keys with a few persistent OUTLIER CHANNELS (massive activations) — the thing that
    breaks naive per-token int-quant."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((seq, heads, dim)).astype(np.float32)
    out_ch = rng.choice(dim, size=max(1, dim // 32), replace=False)
    k[:, :, out_ch] *= rng.uniform(8, 20, size=len(out_ch))
    return k


def make_value(seq, heads, dim, seed=12):
    return np.random.default_rng(seed).standard_normal((seq, heads, dim)).astype(np.float32)


def quant(x, bits, axis):
    """Quantize with scale taken over `axis` (keepdims). axis=2 -> per-token (over feature dim),
    axis=0 -> per-channel (over the token dim). Returns (codes, dequant, scale)."""
    levels = (1 << bits) - 1
    half = levels // 2
    amax = np.abs(x).max(axis=axis, keepdims=True)
    amax[amax == 0] = 1
    scale = amax / half
    q = np.clip(np.round(x / scale).astype(np.int32) + half, 0, levels).astype(np.uint16)
    deq = (q.astype(np.float32) - half) * scale
    return q, deq, scale


def fidelity(orig, recon):
    a = orig.reshape(-1).astype(np.float64)
    b = recon.reshape(-1).astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return cos, rel


def main():
    seq, heads, dim = 2048, 32, 128
    key = make_key(seq, heads, dim)
    n = key.size
    fp16 = n * 2

    print("=== KV-cache compression (seq=%d heads=%d dim=%d) ===" % (seq, heads, dim))
    print("%-36s %8s %9s %9s" % ("method", "vs fp16", "cos-sim", "rel-err"))
    print("%-36s %8s %9s %9s" % ("fp16 (baseline)", "1.00x", "1.000000", "0.0"))

    # KEYS: per-token (wrong) vs per-channel (right)
    for bits in (8, 4):
        for axis, tag in ((2, "per-token (wrong for keys)"), (0, "per-channel (right for keys)")):
            q, deq, scale = quant(key, bits, axis)
            cos, rel = fidelity(key, deq)
            vs = fp16 / (n * bits / 8 + scale.size * 2)
            print("%-36s %7.2fx %9.6f %9.4f" % ("KEY int%d %s" % (bits, tag), vs, cos, rel))

    # range-code the per-channel int4 KEY symbols for extra LOSSLESS shrink
    q, deq, scale = quant(key, 4, axis=0)
    sub = q[:256]
    levels = 16
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
    bpe = len(enc) * 8 / len(flat)
    vs = fp16 / (bpe * n / 8 + scale.size * 2)
    print("%-36s %7.2fx %9s %9s" % ("KEY int4 per-chan + range-coder", vs, "(lossless)", ""))
    print("   (int4 symbols coded at %.3f bits/elem, bit-exact verified)" % bpe)

    # VALUES: per-token is correct and fine
    val = make_value(seq, heads, dim)
    for bits in (8, 4):
        q, deq, scale = quant(val, bits, axis=2)
        cos, rel = fidelity(val, deq)
        vs = (val.size * 2) / (val.size * bits / 8 + scale.size * 2)
        print("%-36s %7.2fx %9.6f %9.4f" % ("VALUE int%d per-token" % bits, vs, cos, rel))

    print("\nTakeaway: on keys, per-token int-quant is wrecked by outlier channels; per-CHANNEL")
    print("fixes it (cos-sim ~1.0 at int8). Values are fine per-token. A range coder shrinks the")
    print("int4 symbols further, losslessly. Real end-task perplexity: bench/validate_ppl_mac.py.")


if __name__ == "__main__":
    main()
