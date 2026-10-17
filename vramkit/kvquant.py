"""Drop-in KV-cache compression for LLM inference — device-agnostic PyTorch (runs on
CUDA, ROCm/AMD, MPS, or CPU). Quantize a key or value tensor, store it small, and
dequantize on the fly in the attention path.

The one thing that matters: KEYS and VALUES want different quantization axes.
Keys carry a few PERSISTENT per-channel outliers (massive-activation dims). A per-TOKEN
scale lets those outliers dominate and crushes every other value — so keys want a
per-CHANNEL scale (one per feature dim, taken across tokens). Values are well-behaved
per token. Getting this right is the difference between +4% and +0.02% perplexity (see
bench/validate_ppl_mac.py — real Qwen2.5-0.5B, measured on a MacBook).

    from vramkit.kvquant import quantize_kv, dequantize_kv
    pk = quantize_kv(key,   bits=8, per="channel")   # keys  -> per-channel  (store this)
    pv = quantize_kv(value, bits=8, per="token")     # values-> per-token
    key_hat   = dequantize_kv(pk)                     # reconstruct in attention
    value_hat = dequantize_kv(pv)

Streaming note: per-channel key scales need the token axis, so in autoregressive decode
you quantize keys in groups of G recent tokens (KIVI-style), keeping the newest few in
fp16. The scheme and numbers here are the quality ceiling that grouping approximates.
"""
from dataclasses import dataclass

try:
    import torch
except ImportError:  # module stays import-safe without torch; functions need it
    torch = None


@dataclass
class PackedKV:
    codes: "torch.Tensor"   # uint8 quantized values (on-wire these pack to `bits` each)
    scale: "torch.Tensor"   # fp16 scales (per-token: one per row; per-channel: one per feature)
    bits: int
    half: int
    per: str                # "token" | "channel"
    token_dim: int          # axis treated as the token axis for per-channel scales


def quantize_kv(t, bits=8, per="token", token_dim=-2):
    """t: (..., dim) key or value tensor. Returns a PackedKV (store this instead of fp16 t).

    per="token"   : scale = absmax over the last (feature) axis — use for VALUES.
    per="channel" : scale = absmax over `token_dim` — use for KEYS (handles their
                    persistent per-channel outliers). token_dim defaults to -2, correct
                    for the canonical (batch, heads, seq, head_dim) cache layout; pass the
                    seq axis explicitly for other layouts.
    """
    if torch is None:
        raise ImportError("quantize_kv needs torch (CUDA/ROCm/MPS/CPU build)")
    half = ((1 << bits) - 1) // 2
    if per == "token":
        amax = t.abs().amax(dim=-1, keepdim=True)
    elif per == "channel":
        amax = t.abs().amax(dim=token_dim, keepdim=True)
    else:
        raise ValueError("per must be 'token' or 'channel'")
    scale = (amax / half).clamp_min(1e-8)
    codes = torch.clamp(torch.round(t / scale) + half, 0, 2 * half).to(torch.uint8)
    return PackedKV(codes=codes, scale=scale.to(torch.float16), bits=bits,
                    half=half, per=per, token_dim=token_dim)


def dequantize_kv(p):
    """Reconstruct the (approximate) fp32 tensor from a PackedKV. The fp16 scale broadcasts
    back whether it was per-token or per-channel."""
    return (p.codes.to(torch.float32) - p.half) * p.scale.to(torch.float32)


def packed_bytes(p):
    """Stored bytes for a PackedKV: `bits` per value (nibble-packed for int4) + fp16 scales."""
    return p.codes.numel() * p.bits / 8 + p.scale.numel() * 2


def self_test():
    assert torch is not None, "need torch"
    torch.manual_seed(0)
    # realistic KV: (tokens, heads, head_dim) with persistent per-channel outliers on keys
    tokens, heads, dim = 512, 8, 128
    key = torch.randn(tokens, heads, dim)
    out_ch = torch.randperm(dim)[:4]
    key[:, :, out_ch] *= 14.0                       # massive-activation channels
    value = torch.randn(tokens, heads, dim)
    fp16 = tokens * heads * dim * 2

    def report(label, t, bits, per):
        p = quantize_kv(t, bits=bits, per=per, token_dim=0)
        r = dequantize_kv(p)
        assert r.shape == t.shape
        cos = torch.nn.functional.cosine_similarity(t.flatten(), r.flatten(), dim=0).item()
        print("%-30s %5.2fx vs fp16   cos-sim %.5f" % (label, fp16 / packed_bytes(p), cos))
        return cos

    print("KEYS (have per-channel outliers):")
    c_tok = report("  int8 per-token  (wrong axis)", key, 8, "token")
    c_ch = report("  int8 per-channel (right axis)", key, 8, "channel")
    report("  int4 per-channel", key, 4, "channel")
    print("VALUES (well-behaved per token):")
    report("  int8 per-token", value, 8, "token")
    report("  int4 per-token", value, 4, "token")
    assert c_ch > c_tok, "per-channel should beat per-token on keys with outliers"
    print("kvquant self_test OK — per-channel wins on keys (device: %s)" % key.device)


if __name__ == "__main__":
    self_test()
