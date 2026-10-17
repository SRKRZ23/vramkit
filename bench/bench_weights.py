"""Weight-codec benchmark: lossless compression of 4-bit quantized LLM weights.
Measures entropy floors, zstd/xz baselines, and our weight-aware range coder (ctx = left+up
neighbours), with bit-exact verification. Realistic weights = low-rank + noise (real transformer
weights are approximately low-rank), per-row absmax int4 quantized."""
import sys, os, subprocess, math, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vramkit.rangecoder import encode_symbols, decode_symbols

def make_weights(rows, cols, rank, seed=7):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((rows, rank)); V = rng.standard_normal((rank, cols))
    W = (U @ V) / math.sqrt(rank)
    W += 0.25 * rng.standard_normal((rows, cols))   # noise floor
    return W.astype(np.float32)

def quantize_int4(W):
    # per-row symmetric absmax to 16 levels (0..15), standard 4-bit quant
    amax = np.abs(W).max(axis=1, keepdims=True); amax[amax == 0] = 1
    q = np.round(W / amax * 7.0).astype(np.int32)     # -7..7
    q = np.clip(q + 8, 0, 15).astype(np.uint8)        # 0..15
    return q

def pack_nibbles(q):
    flat = q.reshape(-1)
    if flat.size % 2: flat = np.append(flat, 0)
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()

def order0_entropy(q):
    h = np.bincount(q.reshape(-1), minlength=16).astype(float); p = h / h.sum()
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())

def cond_entropy_2d(q):
    rows, cols = q.shape
    up = np.zeros_like(q); up[1:, :] = q[:-1, :]
    left = np.zeros_like(q); left[:, 1:] = q[:, :-1]
    ctx = (left.astype(np.int64) * 16 + up.astype(np.int64)).reshape(-1)
    sym = q.reshape(-1).astype(np.int64)
    tab = np.zeros((256, 16), dtype=np.int64)
    np.add.at(tab, (ctx, sym), 1)
    tot = tab.sum()
    H = 0.0
    for c in range(256):
        cs = tab[c].sum()
        if cs == 0: continue
        p = tab[c] / cs
        hc = -(p[p > 0] * np.log2(p[p > 0])).sum()
        H += (cs / tot) * hc
    return float(H)

def cli_size(data, cmd):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data); path = f.name
        out = subprocess.run(cmd + [path], capture_output=True)
        os.unlink(path)
        if cmd[0] == "zstd": return len(out.stdout)
        return len(out.stdout)
    except Exception as e:
        return None

def main():
    rows, cols, rank = 1024, 1024, 64
    W = make_weights(rows, cols, rank)
    q = quantize_int4(W)
    n = q.size
    packed = pack_nibbles(q)
    raw4 = n * 4 / 8  # bytes if stored as raw 4-bit

    H0 = order0_entropy(q)
    H2 = cond_entropy_2d(q)
    zstd = cli_size(packed, ["zstd", "-19", "-c"])
    xz = cli_size(packed, ["xz", "-9e", "-c"])

    # our coder: ctx = left*16 + up  over the 2-D grid
    qflat = q.reshape(-1).tolist()
    def ctx_fn(i, sym, arr):
        r, c = (i + 1) // cols, (i + 1) % cols
        left = arr[(i + 1) - 1] if c > 0 else 0
        up = arr[(i + 1) - cols] if r > 0 else 0
        return left * 16 + up
    enc = encode_symbols(qflat, 256, 16, ctx_fn)
    # verify bit-exact
    dec = decode_symbols(enc, n, 256, 16, ctx_fn)
    assert dec == qflat, "WEIGHT ROUNDTRIP MISMATCH"

    def bpw(nbytes): return nbytes * 8 / n
    print("=== Weight codec benchmark (%dx%d int4, %d weights) ===" % (rows, cols, n))
    print("%-28s %10s %10s %s" % ("method", "bytes", "bits/wt", "vs raw-4bit"))
    for name, b in [("raw 4-bit packed", raw4), ("zstd -19", zstd), ("xz -9e", xz),
                    ("OUR coder (2D ctx)", len(enc))]:
        if b is None: continue
        print("%-28s %10d %10.3f %+.1f%%" % (name, b, bpw(b), (b - raw4) / raw4 * 100))
    print("-- entropy floors --")
    print("order-0 floor:        %.3f bits/wt" % H0)
    print("2-D H(s|left,up) floor: %.3f bits/wt" % H2)
    print("\nour coder vs zstd-19: %+.1f%%  (bit-exact verified: lossless)"
          % ((len(enc) - zstd) / zstd * 100) if zstd else "")
    print("our coder vs raw-4bit: %+.1f%% smaller" % (-(len(enc) - raw4) / raw4 * 100))

if __name__ == "__main__":
    main()
