"""Validate the weight codec on REAL LLM weights (not synthetic).
Parses a bf16 safetensors file manually (no torch needed), quantizes a real weight matrix to
int4, and compares our range coder vs zstd/xz. Run after downloading a model, e.g.:
    python3 -c "from huggingface_hub import hf_hub_download as d; d('Qwen/Qwen2.5-0.5B','model.safetensors',cache_dir='_models')"
    python3 bench/validate_real_weights.py _models/.../model.safetensors"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.bench_weights import quantize_int4, pack_nibbles, order0_entropy, cond_entropy_2d, cli_size
from vramkit.rangecoder import encode_symbols, decode_symbols

def load_bf16_matrix(mf):
    mf = os.path.realpath(mf)
    with open(mf, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(hlen)); base = 8 + hlen
        cand = [(k, v) for k, v in hdr.items() if isinstance(v, dict) and v.get("dtype") == "BF16"
                and len(v.get("shape", [])) == 2 and 512 <= v["shape"][0] <= 2048 and 512 <= v["shape"][1] <= 2048]
        name, meta = cand[0]; o0, o1 = meta["data_offsets"]
        f.seek(base + o0); raw = f.read(o1 - o0)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return name, ((u16.astype(np.uint32)) << 16).view(np.float32).reshape(meta["shape"])

def main():
    mf = sys.argv[1] if len(sys.argv) > 1 else None
    name, W = load_bf16_matrix(mf)
    q = quantize_int4(W); n = q.size; cols = W.shape[1]
    packed = pack_nibbles(q); raw4 = n * 4 / 8
    zstd = cli_size(packed, ["zstd", "-19", "-c"]); xz = cli_size(packed, ["xz", "-9e", "-c"])
    qflat = q.reshape(-1).tolist()
    def ctx_fn(i, sym, arr):
        j = i + 1; left = arr[j - 1] if (j % cols) else 0; up = arr[j - cols] if j >= cols else 0
        return left * 16 + up
    enc = encode_symbols(qflat, 256, 16, ctx_fn)
    assert decode_symbols(enc, n, 256, 16, ctx_fn) == qflat, "MISMATCH"
    bpw = lambda b: b * 8 / n
    print("=== REAL weights: %s (%d weights, int4) ===" % (name, n))
    for nm, b in [("raw 4-bit", raw4), ("zstd -19", zstd), ("xz -9e", xz), ("OUR coder", len(enc))]:
        print("%-16s %10d %7.3f bits/wt  %+.1f%% vs 4bit" % (nm, b, bpw(b), (b - raw4) / raw4 * 100))
    print("OUR vs zstd-19: %+.1f%% | vs raw-4bit: -%.1f%% (lossless, bit-exact)"
          % ((len(enc) - zstd) / zstd * 100, -(len(enc) - raw4) / raw4 * 100))

if __name__ == "__main__":
    main()
