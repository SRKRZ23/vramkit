"""Subbotin-style 32-bit adaptive range coder (pure stdlib).
Encoder and decoder share identical renormalization logic — the #1 correctness trap.
Verified bit-exact on random data before any real use (see self_test)."""

MASK = 0xFFFFFFFF
TOP = 1 << 24
BOT = 1 << 16


class Encoder:
    def __init__(self):
        self.low = 0
        self.rng = MASK
        self.out = bytearray()

    def encode(self, cum, freq, tot):
        self.rng //= tot
        self.low = (self.low + cum * self.rng) & MASK
        self.rng = (self.rng * freq) & MASK
        self._renorm()

    def _renorm(self):
        while True:
            if (self.low ^ (self.low + self.rng)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK

    def finish(self):
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
        return bytes(self.out)


class Decoder:
    def __init__(self, data):
        self.low = 0
        self.rng = MASK
        self.code = 0
        self.data = data
        self.p = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & MASK

    def _byte(self):
        b = self.data[self.p] if self.p < len(self.data) else 0
        self.p += 1
        return b

    def getfreq(self, tot):
        self.rng //= tot
        return (self.code - self.low) // self.rng

    def decode(self, cum, freq, tot):
        self.low = (self.low + cum * self.rng) & MASK
        self.rng = (self.rng * freq) & MASK
        while True:
            if (self.low ^ (self.low + self.rng)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._byte()) & MASK
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK


class AdaptiveModel:
    """Per-context adaptive frequency tables over `nsym` symbols."""
    def __init__(self, nctx, nsym, inc=24):
        self.nsym = nsym
        self.inc = inc
        self.freq = [[1] * nsym for _ in range(nctx)]
        self.tot = [nsym for _ in range(nctx)]

    def cum(self, ctx, sym):
        f = self.freq[ctx]
        return sum(f[:sym]), f[sym], self.tot[ctx]

    def find(self, ctx, val):
        f = self.freq[ctx]
        c = 0
        for s in range(self.nsym):
            if c + f[s] > val:
                return s, c, f[s]
            c += f[s]
        return self.nsym - 1, c - f[-1], f[-1]

    def update(self, ctx, sym):
        self.freq[ctx][sym] += self.inc
        self.tot[ctx] += self.inc
        if self.tot[ctx] >= BOT - 16:
            f = self.freq[ctx]
            t = 0
            for s in range(self.nsym):
                f[s] = (f[s] >> 1) or 1
                t += f[s]
            self.tot[ctx] = t


def encode_symbols(symbols, nctx, nsym, ctx_fn):
    enc = Encoder()
    model = AdaptiveModel(nctx, nsym)
    ctx = 0
    for i, sym in enumerate(symbols):
        cum, freq, tot = model.cum(ctx, sym)
        enc.encode(cum, freq, tot)
        model.update(ctx, sym)
        ctx = ctx_fn(i, sym, symbols)
    return enc.finish()


def decode_symbols(data, n, nctx, nsym, ctx_fn):
    dec = Decoder(data)
    model = AdaptiveModel(nctx, nsym)
    out = []
    ctx = 0
    for i in range(n):
        val = dec.getfreq(model.tot[ctx])
        sym, cum, freq = model.find(ctx, val)
        dec.decode(cum, freq, model.tot[ctx])
        out.append(sym)
        model.update(ctx, sym)
        ctx = ctx_fn(i, sym, out)
    return out


def self_test():
    import random
    rnd = random.Random(12345)
    # order-1 context over 16 symbols, biased distribution
    n = 20000
    syms = []
    prev = 0
    for _ in range(n):
        # biased, context-dependent draw
        r = rnd.random()
        if r < 0.4:
            s = prev
        elif r < 0.7:
            s = rnd.randint(0, 3)
        else:
            s = rnd.randint(0, 15)
        syms.append(s)
        prev = s
    ctx_fn = lambda i, sym, arr: sym  # context = previous symbol
    enc = encode_symbols(syms, 16, 16, ctx_fn)
    dec = decode_symbols(enc, n, 16, 16, ctx_fn)
    assert dec == syms, "ROUNDTRIP MISMATCH"
    print("self_test: bit-exact roundtrip OK on %d random symbols, %d bytes (%.3f bits/sym)"
          % (n, len(enc), len(enc) * 8 / n))
    return True


if __name__ == "__main__":
    self_test()
