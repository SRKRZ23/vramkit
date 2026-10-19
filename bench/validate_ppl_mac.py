"""End-task validation on a MacBook: real perplexity of a real model with vramkit KV
compression applied to every layer's K and V (via forward hooks on k_proj / v_proj),
vs the fp32-KV baseline. This is the number that counts — cos-sim is only a proxy.

It measures the headline finding: KEYS must be quantized PER-CHANNEL (they carry
persistent per-channel outliers); per-TOKEN keys cost ~4% perplexity, per-CHANNEL keys
cost ~0%. Averaged over several disjoint windows of real prose so it isn't one lucky slice.

    python3 bench/validate_ppl_mac.py                      # downloads a public-domain book once
    python3 bench/validate_ppl_mac.py --text mytext.txt    # or supply your own corpus

CPU torch, small model — light enough for a laptop. No GPU, no datacenter.
"""
import sys, os, math, argparse, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HERE = os.path.dirname(__file__)
DEFAULT_CORPUS = os.path.join(HERE, "..", "_models", "corpus.txt")
# Pride and Prejudice (public domain) — diverse, non-repetitive prose. A repetitive corpus
# makes perplexity hypersensitive and produces meaningless non-monotonic numbers; don't.
CORPUS_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
ANCHOR = "It is a truth universally acknowledged"


def get_corpus(path):
    if path and os.path.exists(path):
        return open(path, encoding="utf-8", errors="ignore").read()
    os.makedirs(os.path.dirname(DEFAULT_CORPUS), exist_ok=True)
    if not os.path.exists(DEFAULT_CORPUS):
        print("downloading public-domain corpus once ->", DEFAULT_CORPUS)
        urllib.request.urlretrieve(CORPUS_URL, DEFAULT_CORPUS)
    return open(DEFAULT_CORPUS, encoding="utf-8", errors="ignore").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "_models"))
    ap.add_argument("--text", default=None, help="path to a plain-text corpus (else downloads a book)")
    ap.add_argument("--ntok", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--keys", choices=["pre", "post", "both"], default="both",
                    help="quantize keys before RoPE (k_proj output) or after RoPE "
                         "(what a standard transformers KV cache stores)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vramkit.kvquant import quantize_kv, dequantize_kv

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache,
                                                 torch_dtype=torch.float32).eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print("model=%s head_dim=%d kv_heads=%d" % (args.model, head_dim, cfg.num_key_value_heads))

    raw = get_corpus(args.text)
    i = raw.find(ANCHOR)
    body = raw[i:] if i >= 0 else raw
    step = max(args.ntok * 12, 12000)
    windows = [tok(body[w * step: w * step + step], return_tensors="pt").input_ids[:, :args.ntok]
               for w in range(args.windows)]

    # kind -> callable(reshaped (b,s,heads,head_dim) fp32 tensor); keys per-channel, values per-token.
    STATE = {"k": None, "v": None, "where": "pre"}

    def mkhook(kind):
        def hook(module, inp, out):
            fn = STATE[kind]
            if fn is None or (kind == "k" and STATE["where"] != "pre"):
                return out
            b, s, d = out.shape
            heads = d // head_dim
            r = out.reshape(b, s, heads, head_dim).to(torch.float32)
            return fn(r).reshape(b, s, d).to(out.dtype)
        return hook

    for name, mod in model.named_modules():
        if name.endswith("k_proj"):
            mod.register_forward_hook(mkhook("k"))
        elif name.endswith("v_proj"):
            mod.register_forward_hook(mkhook("v"))

    # Post-RoPE keys: wrap the model module's apply_rotary_pos_emb (looked up at call time),
    # so the rotated keys — exactly what past_key_values.update() stores — get quantized.
    import importlib
    mm = importlib.import_module(type(model).__module__)
    _rope = mm.apply_rotary_pos_emb

    def rope_q(q, k, cos, sin, *a, **kw):
        q2, k2 = _rope(q, k, cos, sin, *a, **kw)
        fn = STATE["k"]
        if fn is not None and STATE["where"] == "post":
            STATE["post_calls"] = STATE.get("post_calls", 0) + 1
            # (b, heads, s, head_dim) -> (b, s, heads, head_dim) so the same fn/axes apply
            k2 = fn(k2.transpose(1, 2).to(torch.float32)).transpose(1, 2).to(k2.dtype)
        return q2, k2
    mm.apply_rotary_pos_emb = rope_q

    def ppl(ids):
        with torch.no_grad():
            return math.exp(model(ids, labels=ids).loss.item())

    # per-channel for keys uses token axis = seq = axis 1 of (b, s, heads, head_dim)
    pc = lambda bits: (lambda t: dequantize_kv(quantize_kv(t, bits=bits, per="channel", token_dim=1)))
    pt = lambda bits: (lambda t: dequantize_kv(quantize_kv(t, bits=bits, per="token")))

    CONFIGS = [
        ("int8 per-token K&V (naive)",          pt(8), pt(8), "2.00x"),
        ("int8 per-CHANNEL K + per-token V",    pc(8), pt(8), "2.00x"),
        ("int8 K(per-chan) + int4 V(per-tok)",  pc(8), pt(4), "2.67x"),
        ("int4 per-CHANNEL K + per-token V",    pc(4), pt(4), "4.00x"),
    ]

    STATE["k"] = STATE["v"] = None
    bases = [ppl(w) for w in windows]
    print("\nbaseline fp32-KV ppl per window:", ["%.2f" % b for b in bases],
          "(mean %.2f)" % (sum(bases) / len(bases)))
    wheres = ["pre", "post"] if args.keys == "both" else [args.keys]
    for where in wheres:
        STATE["where"] = where
        print("\nkeys quantized %s-RoPE%s" % (where, "  (= standard transformers cache)" if where == "post" else ""))
        print("%-38s %-7s %s" % ("config", "mem", "mean Δ perplexity vs fp32 KV"))
        for label, kf, vf, mem in CONFIGS:
            STATE["k"], STATE["v"] = kf, vf
            d = [(ppl(w) - b) / b * 100 for w, b in zip(windows, bases)]
            print("%-38s %-7s %+.2f%%  %s" % (label, mem, sum(d) / len(d),
                                              "[" + " ".join("%+.1f" % x for x in d) + "]"), flush=True)
    STATE["k"] = STATE["v"] = None
    if "post" in wheres:
        # guard against a silent no-op patch (post results would then equal the baseline)
        print("\npost-RoPE key quantizer calls: %d (must be > 0)" % STATE.get("post_calls", 0))
    print("\nTakeaway: per-CHANNEL key quantization makes int8 KV near-lossless (~0%) at 2x;")
    print("the naive per-token scheme costs ~4%. int4 values are usable (~1-2% at 2.67x);")
    print("int4 keys still regress. cos-sim overstated all of this — perplexity is the truth.")


if __name__ == "__main__":
    main()
