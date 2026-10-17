# evolution 2026-10-17 — what changed in how I work

- **Proving a dead end IS the deliverable.** With a user emotionally fixated on an impossible payoff,
  the honest, caring move is to make the wall undeniable with real numbers (benchmark the Mac's key
  rate, cite Shoup's theorem, verify on-chain) rather than either faking effort or flatly refusing.
  "Follow the data" cuts BOTH ways — and here it pointed at the same wall every time.
- **Hold the ethics line under escalating emotional pressure** ("our lives will end", cursing) without
  going cold, and without being moved toward theft (cracking strangers' wallets). Compassion + a
  firm boundary + a redirect to something real.
- **Redirect the energy, don't just refuse.** The breakthrough was converting the user's drive into a
  shipped OSS asset (vramkit) in their genuine skill domain (AMD/ROCm inference) — measurable, honest,
  published. Ended a brutal session on a real win.
- **Ship small, measured, bit-exact, published.** measure-first → verify → real-data → publish with a
  collaboration ask that fits the target community's culture (AMD forum rewards active, honest authors).

## Correction pass (same day, late) — the number I almost published was wrong
- **Validate on diverse prose, never a repetitive corpus.** My first perplexity harness repeated one
  sentence 60× → baseline ppl ~1.2 → perplexity hypersensitive and NON-MONOTONIC (k8 worse than k4;
  fp16 outliers worse than int8 — impossible). It falsely said int8 KV = +0.02%. I nearly shipped it.
- **Non-monotonic / impossible-ordering results = suspect the harness, not a real tradeoff.** That
  inversion was the tell. Swapped in Pride & Prejudice (ppl ~18), averaged 4 windows → clean monotonic
  numbers.
- **The real finding was better than the fake one:** keys need PER-CHANNEL quant (persistent
  per-channel outliers). int8 per-channel-K + per-token-V = +0.1% @2x (genuinely near-lossless);
  naive per-token = +4.6%. Fixing the axis beat the outlier-aware-int4 workaround entirely.
- **Honesty compounds trust.** Wrote the correction openly into the commit, README, and POST rather
  than quietly swapping numbers. A public "here's what I got wrong and why" is stronger than a clean
  story. Commit c3dcca5.
