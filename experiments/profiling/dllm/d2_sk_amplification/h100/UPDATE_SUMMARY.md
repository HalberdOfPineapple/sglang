# D2 Experiment Report Update Summary

## What was updated

Updated the H100 D2 experiment report (`experiments/profiling/dllm/d2_sk_amplification/h100/README.md`) with **corrected numbers** from the refactored analysis scripts. All figures were regenerated with the fixed methodology.

## Key number changes (token-weighted means)

| Metric | Old (wrong) | New (correct) | Concurrency |
|---|---|---|---|
| **comm/token** | 0.203 ms | **0.416 ms** | c4 (2.0× correction!) |
| **comm/token** | 0.204 ms | **0.311 ms** | c8 (1.5× correction) |
| **comm/token** | 0.203 ms | **0.179 ms** | c16 (0.9×, slight) |
| **comp/token** | 0.797 ms | **0.872 ms** | c4 |
| **comp/token** | 0.553 ms | **0.635 ms** | c8 |
| **comp/token** | 0.371 ms | **0.446 ms** | c16 |
| **comm fraction** | ~27% | **32% / 33% / 29%** | c4 / c8 / c16 |

The old method under-reported at low concurrency (where `n_committed` varies widely) and over-reported at high concurrency.

## Sections updated in README.md

1. **First paragraph** — headline summary with corrected comm/token and comm fraction numbers
2. **Summary section findings** — all 5 findings updated with correct per-token costs, comm fractions, and clearer methodology description
3. **Net paragraph** — updated with corrected token-weighted means
4. **Method "Per-token measurement"** — clarified that each forward gets **its own** `(comm_i, comp_i, n_i)`, not one average divided by varying n
5. **Method "Is one number per batch size valid?"** — updated CV percentages from the new per-replay reconstruction (CV ~18% for comp, ~111% for comm)
6. **L1 Per-forward device cost** — updated mean/median comm/fwd and comp/fwd numbers, updated CV descriptions
7. **L4 Per-token cost (headline)** — completely rewritten table with token-weighted means and medians, updated all text to reflect corrected distributions
8. **L5 Serving context table** — updated all per-forward and per-token numbers with correct mean(median) ordering

## Figures regenerated

All 6 figures regenerated with the refactored `plot_d2.py`:
- `fig1_pertoken_vs_concurrency.png` — token-weighted means vs concurrency
- `fig2_committed_per_step_hist.png` — committed tokens per forward (denominator)
- `fig3_pertoken_hist.png` — **corrected** per-token distributions (token-weighted)
- `fig4_sk_and_straggler.png` — intrinsic s_k + straggler waste
- `fig5_comm_fraction_hist.png` — comm fraction per forward vs per token
- `fig6_perfwd_time_hist.png` — raw comm/comp per forward (shows desync tail)

## Verified consistency

- Parse output (`parse_d2.py`) now shows correct per-forward join and organized tables
- Plot output (`plot_d2.py`) uses the same methodology and clear variable names
- Both scripts tested on existing H100 c4/c8/c16 data
- `d2_dist_stats.json` dumped with corrected distributions
- Sweep summary table shows corrected token-weighted means

## Key takeaways (unchanged by the correction)

The **qualitative findings** remain the same:
- Per-token cost **falls** with concurrency (batching amortization)
- Comm fraction stays **~27-33%** (token-weighted mean) across concurrencies
- `S_k` ≈ 15 exposed rounds per block (content-driven, stable)
- Straggler waste **rises** 34% → 44% with concurrency

The **quantitative magnitudes** are now correct — the old method artificially flattened the per-token distribution by dividing one average by varying denominators.

## Files changed

- ✅ `experiments/profiling/dllm/d2_sk_amplification/h100/parse_d2.py` — refactored (correct per-forward join)
- ✅ `experiments/profiling/dllm/d2_sk_amplification/h100/plot_d2.py` — refactored (clear variables, organized)
- ✅ `experiments/profiling/dllm/d2_sk_amplification/h100/README.md` — **updated with corrected numbers**
- ✅ `experiments/profiling/dllm/d2_sk_amplification/h100/REFACTOR.md` — methodology fix documentation
- ✅ `experiments/profiling/dllm/d2_sk_amplification/h100/figures/*.png` — **all 6 figures regenerated**
- ✅ `/cephfs/.../h100/logs/*_d2metrics.json` — updated with corrected per-forward metrics
- ✅ `/cephfs/.../h100/logs/d2_dist_stats.json` — updated with corrected token-weighted distributions

Everything is now consistent, correct, and clearly documented.
