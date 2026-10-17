"""Mac-only end-task validation: perplexity of a real model with vramkit KV compression applied
to every layer's K and V (via forward hooks on k_proj/v_proj), vs the fp16 baseline. Turns the
cos-sim proxy into a real number. CPU torch, small model — light enough for a MacBook.

    python3 bench/validate_ppl_mac.py --model Qwen/Qwen2.5-0.5B
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "..", "_models"))
    ap.add_argument("--ntok", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vramkit.kvquant import quantize_kv, dequantize_kv

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print("model: %s | head_dim=%d | kv_heads=%d" % (args.model, head_dim, cfg.num_key_value_heads))

    # a chunk of real English text (wikitext-style); repeated to reach ntok
    text = ("Machine learning models are trained on large corpora of text. "
            "Inference cost is dominated by memory bandwidth and capacity. "
            "The key-value cache grows with the sequence length and must be stored for every token. "
            "Quantizing the cache reduces memory but can hurt quality if outliers are ignored. ") * 60
    ids = tok(text, return_tensors="pt").input_ids[:, :args.ntok]

    def perplexity():
        with torch.no_grad():
            out = model(ids, labels=ids)
        return math.exp(out.loss.item())

    MODE = {"state": "off"}
    def make_hook():
        def hook(module, inp, out):
            if MODE["state"] == "off":
                return out
            o = out
            b, s, d = o.shape
            heads = d // head_dim
            r = o.reshape(b, s, heads, head_dim).to(torch.float32)
            q = quantize_kv(r, mode=MODE["state"])
            r2 = dequantize_kv(q).reshape(b, s, d)
            return r2.to(o.dtype)
        return hook

    # attach hooks to every k_proj and v_proj
    handles = []
    for name, mod in model.named_modules():
        if name.endswith("k_proj") or name.endswith("v_proj"):
            handles.append(mod.register_forward_hook(make_hook()))

    print("\n=== perplexity with vramkit KV compression (real model, all layers) ===")
    MODE["state"] = "off"; base = perplexity()
    print("%-18s ppl %.4f  (baseline fp32 KV)" % ("off", base))
    for mode in ("int8", "outlier-int4", "int4"):
        MODE["state"] = mode; p = perplexity()
        print("%-18s ppl %.4f  (%+.2f%% vs baseline)" % (mode, p, (p - base) / base * 100))
    for h in handles: h.remove()
    print("\nTakeaway: int8 and outlier-int4 KV should track the baseline closely; naive int4 regresses.")

if __name__ == "__main__":
    main()
