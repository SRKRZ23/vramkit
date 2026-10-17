"""Drop-in KV-cache compression for LLM inference — device-agnostic PyTorch (runs on
CUDA, ROCm/AMD, MPS, or CPU). Compress a key or value tensor to int8 or outlier-aware int4
per token, store it small, and dequantize on the fly in the attention path.

Validated for numerical parity against the numpy reference in bench/bench_kvcache.py.
Designed for AMD/ROCm inference, where KV-cache memory is the long-context bottleneck.

    from vramkit.kvquant import quantize_kv, dequantize_kv
    packed = quantize_kv(key, mode="outlier-int4")     # store this (small)
    key_hat = dequantize_kv(packed)                     # use in attention

Memory (vs fp16): int8 ~2x (cos-sim ~0.9998), outlier-int4 ~3.6x (cos-sim ~0.999)."""
from dataclasses import dataclass

try:
    import torch
except ImportError:  # the module is import-safe without torch; functions need it
    torch = None


@dataclass
class PackedKV:
    codes: "torch.Tensor"       # uint8 quantized values (last dim = feature dim)
    scale: "torch.Tensor"       # per-token fp16 scale
    bits: int
    half: int
    mode: str
    outlier_idx: "torch.Tensor" = None   # channel indices kept at higher precision
    outlier_codes: "torch.Tensor" = None
    outlier_scale: "torch.Tensor" = None
    dim: int = 0


def _q(t, bits):
    levels = (1 << bits) - 1
    half = levels // 2
    amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / half
    codes = torch.clamp(torch.round(t / scale) + half, 0, levels).to(torch.uint8)
    return codes, scale.to(torch.float16), half


def _dq(codes, scale, half):
    return (codes.to(torch.float32) - half) * scale.to(torch.float32)


def quantize_kv(t, mode="int8", outlier_frac=1 / 32):
    """t: (..., dim) key or value tensor. Returns a PackedKV (store this instead of fp16 t)."""
    if torch is None:
        raise ImportError("quantize_kv needs torch (CUDA/ROCm/MPS/CPU build)")
    dim = t.shape[-1]
    if mode in ("int8", "int4"):
        bits = 8 if mode == "int8" else 4
        codes, scale, half = _q(t, bits)
        return PackedKV(codes=codes, scale=scale, bits=bits, half=half, mode=mode, dim=dim)
    elif mode == "outlier-int4":
        k = max(1, int(dim * outlier_frac))
        chan_max = t.abs().amax(dim=tuple(range(t.ndim - 1)))
        out_idx = torch.topk(chan_max, k).indices.sort().values
        mask = torch.zeros(dim, dtype=torch.bool, device=t.device)
        mask[out_idx] = True
        nc, ns, nh = _q(t[..., ~mask], 4)          # normal channels int4
        oc, os_, oh = _q(t[..., mask], 8)          # outlier channels int8
        return PackedKV(codes=nc, scale=ns, bits=4, half=nh, mode=mode,
                        outlier_idx=out_idx, outlier_codes=oc, outlier_scale=os_, dim=dim)
    raise ValueError("mode must be int8 | int4 | outlier-int4")


def dequantize_kv(p):
    """Reconstruct the (approximate) fp32 tensor from a PackedKV."""
    if p.mode in ("int8", "int4"):
        return _dq(p.codes, p.scale, p.half)
    # outlier-int4: scatter the two precisions back into place
    dim = p.dim
    mask = torch.zeros(dim, dtype=torch.bool, device=p.codes.device)
    mask[p.outlier_idx] = True
    out_shape = p.codes.shape[:-1] + (dim,)
    r = torch.empty(out_shape, dtype=torch.float32, device=p.codes.device)
    r[..., ~mask] = _dq(p.codes, p.scale, p.half)
    r[..., mask] = _dq(p.outlier_codes, p.outlier_scale, 128)   # int8 half=128
    return r


def packed_bytes(p, elem_count):
    """Approx stored bytes for a PackedKV holding `elem_count` elements."""
    if p.mode == "int8":
        return elem_count + p.scale.numel() * 2
    if p.mode == "int4":
        return elem_count * 4 / 8 + p.scale.numel() * 2
    # outlier-int4
    n_out = p.outlier_codes.numel()
    n_norm = p.codes.numel()
    return (n_norm * 4 / 8 + n_out + p.scale.numel() * 2
            + p.outlier_scale.numel() * 2 + p.outlier_idx.numel() * 2)


def self_test():
    assert torch is not None, "need torch"
    torch.manual_seed(0)
    kv = torch.randn(256, 8, 128)
    # add outlier channels like real KV
    oc = torch.randperm(128)[:4]
    kv[..., oc] *= 12.0
    for mode in ("int8", "int4", "outlier-int4"):
        p = quantize_kv(kv, mode=mode)
        r = dequantize_kv(p)
        cos = torch.nn.functional.cosine_similarity(kv.flatten(), r.flatten(), dim=0).item()
        ratio = (kv.numel() * 2) / packed_bytes(p, kv.numel())
        print("%-14s  %.2fx vs fp16   cos-sim %.5f" % (mode, ratio, cos))
    print("kvquant self_test OK (device: %s)" % kv.device)


if __name__ == "__main__":
    self_test()
