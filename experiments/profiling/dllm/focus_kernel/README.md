# Experiment F5 — FOCUS Phase-B Triton kernels ON vs OFF (LLaDA2.0-mini, 1×A100 80GB, TP=1, eager)

Measures the wall-clock effect of the FOCUS Phase-B Triton kernels (`§B1` importance + `§B2` selection, `algorithm/focus_kernels.py`, gated by `SGLANG_FOCUS_KERNEL=1`) against the PyTorch-oracle path (`SGLANG_FOCUS_KERNEL` unset), FOCUS-vs-FOCUS on the same reduced forward. This is the follow-up the F1/F3-lite writeups called for: F3-lite attributed `select ~14%` (host Python per-request selection loop) + an importance einsum inside `p_fwd` to pure host cost; F5 confirms the kernels remove it and quantifies the throughput win.

## Headline
**The kernels make FOCUS faster, and the win grows with batch — 1.04× / 1.23× / 1.41× at conc 1 / 8 / 16.** At conc-16 kernel-FOCUS reaches **395 tok/s**, drawing level with the original LowConfidence baseline (388 tok/s, F1) — i.e. the §B kernels close the entire eager host-overhead gap that made prototype FOCUS 0.69× at conc-16. **Redundancy is unchanged** (0.66/0.80/0.80 both paths): the kernels change *how* importance/selection are computed, not *which* tokens are evicted, so this is a pure host-efficiency win with identical model behavior (and the α→∞ anchor stays bit-identical to eager — see `focus_a100_smoke`).

## Throughput (mode T — no phase-timing syncs)
| conc | OFF tok/s | ON tok/s | speedup | OFF lat (mean) | ON lat | redund OFF/ON (mean) |
|---|---|---|---|---|---|---|
| 1  | 61  | 64  | **1.04×** | 2.09s | 2.01s | 0.662 / 0.661 |
| 8  | 232 | 285 | **1.23×** | 4.32s | 3.41s | 0.796 / 0.792 |
| 16 | 281 | 395 | **1.41×** | 6.70s | 4.67s | 0.807 / 0.799 |

The speedup scales with concurrency because the removed cost was **O(bs) host work per denoising step** (the Python per-request selection loop and the per-request importance einsum): at bs=1 there is little to amortize (+4%), at bs=16 it dominated the exposed host time (+41%). Redundancy is statistically identical OFF vs ON at every point (Δ ≤ 0.008, within run-to-run noise), confirming the eviction decision is preserved.

## vs LowConfidence (same-session baseline — the F1 question, re-answered)
F1 found prototype FOCUS was **0.90 / 0.84 / 0.69×** LowConfidence (a loss, widening with batch). To
compare cleanly against the kernel path I re-ran LowConfidence in **this** session/harness
(`run_lowconf_baseline.sh`, identical driver/config/box state — avoids F1's cross-session drift):

| conc | LowConf | FOCUS-oracle | FOCUS-kernel | oracle/LC | **kernel/LC** |
|---|---|---|---|---|---|
| 1  | 66  | 61  | 64  | 0.93× | **0.96×** |
| 8  | 276 | 232 | 285 | 0.84× | **1.03×** |
| 16 | 375 | 281 | 395 | 0.75× | **1.05×** |

**The kernels flip FOCUS from a loss into a win/parity vs LowConfidence.** Oracle FOCUS reproduces F1
(0.93/0.84/0.75×, losing more as batch grows); kernel FOCUS is **0.96 / 1.03 / 1.05×** — it **beats
LowConfidence at conc 8 and 16** and is near-parity at conc-1, and the advantage now *grows* with batch
(the opposite direction from the prototype). Latency crosses over too: at conc 8/16 kernel-FOCUS mean
latency (3.41 / 4.67s) is below LowConfidence (3.59 / 4.89s). This is the first configuration where the
paper's FLOPs eviction turns into an actual wall-clock win over the stock path in this eager small-MoE
regime — because FOCUS does the same GPU MoE work on ~20% fewer tokens (redundancy 0.66–0.80) while the
§B kernels remove the host overhead that previously masked that saving. §C (Phase-S graph) would widen
the margin further (it attacks the 67.5% `s_fwd` share that FOCUS and LowConfidence both pay).

## Per-phase attribution (mode P — `SGLANG_FOCUS_PHASE_TIMING=1`, conc 8)
Phase timing syncs at phase boundaries, so absolute ms are inflated and **shares are the signal**. OFF vs ON, aggregated over 24 blocks:

| phase | OFF ms | OFF % | ON ms | ON % | note |
|---|---|---|---|---|---|
| s_fwd | 10463 | 56.1% | 10298 | **67.5%** | Phase S = L2..L (18 MoE layers) on \|S\| — unchanged work, now a *bigger* share |
| p_fwd | 3235 | 17.4% | 1829 | **12.0%** | §B1: importance einsum → Triton kernel + dropped `repeat_interleave` (−43% ms) |
| select | 2253 | 12.1% | 363 | **2.4%** | §B2: host Python selection loop → one Triton launch (**−84% ms, 6.2×**) |
| a1_fwd | 1306 | 7.0% | 1333 | 8.7% | unchanged (L1 attn on \|S\|) |
| commit | 427 | 2.3% | 483 | 3.2% | unchanged |
| s/a1/p_meta | 955 | 5.2% | 943 | 6.2% | unchanged (3× `init_forward_metadata`) |
| **total** | **18639** | | **15249** | | **−18% wall (with sync inflation)** |

`select` collapsed **12.1% → 2.4%** (the §B2 target) and `p_fwd` dropped **17.4% → 12.0%** (§B1 kernel + no kv-head broadcast). Together the two kernels cut host-phase share from **29.5% → 14.4%** of wall. `s_fwd` (the Phase-S MoE forward) is unchanged in absolute ms but rises to **67.5%** of the (now smaller) total — with the host cost removed, the eager per-layer launch latency of L2..L is even more clearly the remaining lever, reconfirming **§C (Phase-S CUDA graph)** as the next target.

## Setup
- **HW/SW:** 1× A100 80GB PCIe (sm80), TP=1, conda env `sglang`, FlashInfer, eager (`--disable-cuda-graph --attention-backend flashinfer`). Working tree = commit `9c72fe320` + the session-4 Phase-B kernel changes (`focus_kernels.py`, wiring in `focus.py`/`llada2.py`/`focus_utils.py`).
- **Model:** `/cephfs/shared/model/LLaDA2.0-mini` (llada2_moe, 20 layers, hidden 2048, 256 experts/top-8, bf16); `block_size=32`. FOCUS α=1.5, thr 0.9, maxpool_k 3, importance_layers [0,1].
- **Workload:** HumanEval (first 20 prompts cycled), sustained concurrency ∈ {1,8,16} = `max_running_requests`, `total_reqs=3×conc`, `max_new_tokens=128` (4 blocks of 32), greedy. One unmeasured warmup per run. Reuses the F1 driver (`../focus_vs_lowconf/drive_humaneval.py`) and `configs/focus.yaml`.
- **Modes:** (T) throughput, no phase timing → tok/s; (P) phase split, timing ON → shares. Kept separate because the timing syncs distort tok/s. Redundancy CSV logged in T to confirm identical eviction.

## Method & tooling
- `SGLANG_FOCUS_KERNEL=0|1` selects oracle vs Triton path; `SGLANG_FOCUS_PHASE_TIMING=1` prints per-block `[focus-timing]` (mode P); `SGLANG_FOCUS_REDUNDANCY_CSV` logs Σ\|S\|/(B·bs) per step (mode T). Bulletproof server teardown (`kill_servers` waits for GPU < 2 GB) before each run so no leftover server races the port/env.
- **Reproduce:**
  ```bash
  cd /root/sglang_a100/sglang/experiments/profiling/dllm/focus_kernel
  ./run_focus_kernel.sh                     # T sweep {1,8,16} × {OFF,ON} + P at conc 8
  python parse_focus_kernel.py <LOGS_DIR>    # table above
  ```

## Artifacts
Data (mirror, not in repo): `/cephfs/shared/wxli/sglang-dllm/profiling/dllm/focus_kernel/logs/` — `k{0,1}_c{conc}_{T,P}_{result.json,server.log,drive.log,redundancy.csv}`, `focus_kernel_summary.txt`. Scripts (repo, this dir): `run_focus_kernel.sh`, `parse_focus_kernel.py`.

## Caveats
- Small MoE, eager, single GPU — the same regime as F1. The kernels help here precisely because the regime is host-bound; a compute-bound regime (bigger model / longer ctx) would show a smaller *relative* host share but the kernels never hurt (redundancy-neutral, strictly less host work).
- Phase-P absolute ms retains one D2H (`new_lens_cpu`) and the 3× metadata rebuild — untouched by §B, and correctly small (~5%). The residual `p_fwd` (12%) is now mostly the genuine L0-full + L1-QKV block forward, not the importance side-channel.
- Run-to-run tok/s noise is ±few % (single short measured window); the monotone 1.04→1.23→1.41 trend and the `select` 12%→2.4% collapse are well outside it.
