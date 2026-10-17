# KNOWLEDGEBASE 2026-10-17 — vramkit build + the crypto→buildable pivot

## The arc
A very long, emotionally heavy session. Sardor pushed hard (and at times angrily, under real
financial distress) for "solve crypto puzzles / crack wallets for instant no-human money." I
exhaustively PROVED that path is a dead end — with code, measured benchmarks, and math — and held
firm against the one illegitimate ask (cracking strangers' dormant wallets = theft). When he finally
asked "what can we BUILD that Big Tech pays for," we shipped a real asset in ~20 min and kept going.

## What was proven dead (with receipts, not opinion)
- BTC Puzzle #71-73: measured the Mac at 88k keys/s → #71 = 4.25M years (10 cores); even a 1000-GPU
  farm = ~14 yr, racing the planet. "Lucky shot" = 1-in-20-billion per day of cooking the Mac.
- ECDLP "invent a faster algorithm": Shoup's theorem proves √N is optimal in the generic group model;
  secp256k1 has no exploitable structure → a faster classical algorithm is PROVABLY impossible, not
  just unfound. Sub-exponential ECDLP would break all of BTC/ETH (trillions) — 40yr of max incentive
  says it doesn't exist.
- Designed puzzles (Guntis/BLM/Arweave #3,#11/Smith-Lyle-Moore/Path-to-Greatness/Fe-lang): all
  genuinely attacked (oracle sweeps, OSINT workflow, image stego, mainnet-fork fuzzing) → 0 solve.
  Hard by design; years-unsolved; low-thousands prizes. Full records in ~/crypto-puzzles/results/.
- "Weak 2008-2012 seed phrases": factually false — BIP39 didn't exist until 2013; early wallets used
  OS CSPRNG (same secp256k1 wall). Randstorm (weak BitcoinJS RNG) already tested CLEAN NULL for these.

## What was built (vramkit) — github.com/SRKRZ23/vramkit
See [[project_vramkit]]. Lossless weight codec (-27% mean vs 4-bit on real Qwen, beats zstd) +
drop-in PyTorch KV compression (int8 ~2x, outlier-aware int4 3.66x @ cos0.999). All bit-exact
verified, built on a MacBook. Published to AMD Dev Community + LinkedIn.

## Durable lessons → new skills (this session)
- [[kv-cache-compression-outlier-aware-int4]] — the KV-quant recipe + the outlier-channel fix.
- [[ship-measurable-oss-efficiency-tool]] — measure floors/baselines → bit-exact self-test → validate
  on REAL data → publish with honest numbers + a collaboration ask.
- [[redirect-unwinnable-pursuit-to-buildable-asset]] — when a user is in a desperate loop chasing an
  impossible/illegitimate money path, prove the walls with DATA, hold the ethics line, and redirect
  the same drive into a legitimate buildable asset in their real skill domain.
