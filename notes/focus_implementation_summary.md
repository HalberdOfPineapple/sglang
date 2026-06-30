# FOCUS Implementation Summary

**Branch:** `feature/focus-implementation`
**Updated:** 2026-06-30 (session 3d)
**Status:** Paper-exact reduced forward COMPLETE + validated; host de-sync (Plan-A §A) shipped; CUDA-graph (§C) mechanism proven but naive integration dead-ended (fixed-shape rework pending).

This is a current snapshot. The dated change-log lives in `notes/focus_implementation_progress.md`; the optimization plan in `notes/focus_graph_kernel_plan.md`; the §C design in `notes/focus_phase_s_graph_design.md`.

## Where it stands
FOCUS is implemented end-to-end as a paper-exact **reduced (token-evicting) forward** and runs on a single A100 with LLaDA2.0-mini. Each denoising step physically evicts non-decodable tokens after Layer 1, so Layers 1-attn..L execute on `|S| ≪ B` tokens (the real FLOPs lever). Correctness is anchored by **α→∞ ≡ LowConfidence** (bit-for-bit). The remaining work is turning the measured FLOPs cut into a wall-clock win.

## Implemented + validated
### Core algorithm (`algorithm/focus.py`)
- 3-phase split per step: **P** (embed→L0 full→L1 QKV+RoPE+full-block KV fill, collect I0/I1) → host **select+compact** → **A1** (L1 attn on |S| vs full-block KV) → **S** (L2..L on |S|, KV→block prefix, read context+|S|) → commit confident retained-masked positions. Forces the paged FlashInfer path.
- Budget/selection match the official kernels: `compute_focus_targets` (no N_σ in budget), `compute_should_evict`, `select_and_enforce_constraints` (top-target OR N_σ mean+std expansion, AR-context adjacency, placeholder progress-gate). PyTorch reference impls kept as numerical oracles for the eventual Triton port.
- Per-request cumulative decode stats drive the dynamic budget K (Eq. 17-19), kept on-device (§A4).

### Model split (`models/llada2.py`)
`LLaDA2MoeAttention.forward_qkv_rope` / `.write_kv` / `.forward_attn`; block-level `forward_focus_prefix_attn` / `forward_focus_suffix`; model-level `forward_focus_prefix` / `forward_focus_l1_suffix` / `forward_focus_rest`; LM delegators incl. `forward_focus_rest_and_logits`. The monolithic `forward` is untouched (normal serving unaffected). Importance side-channel (`FocusRuntimeView` on `ForwardBatch.focus_view`) collects I0/I1 at layers 0/1.

### Split-forward metadata (`algorithm/focus_forward.py`, `focus_reduce.py`)
Per-phase `seq_lens`/`extend_*` builders (`compute_focus_phase_lens`, `make_focus_phase_batch`), contiguous block-prefix KV slots (`build_phase_s_out_cache_loc`), on-device compaction (`build_retained_index_from_mask`). Works with NO custom kernel: SGLang's FlashInfer dLLM-extend uses token-granular `kv_indices` from a contiguous `req_to_token` slice, so a phase is selected purely by its `seq_lens`/`prefix_lens` given retained KV in a contiguous block prefix.

### Plan-A §A — host-path de-sync (SHIPPED, behavior-preserving)
Reduced the per-step O(bs) D2H syncs to O(1): batched `_select_retained`, on-device index builders, single-D2H `make_focus_phase_batch`, fully-vectorized ragged `_commit_step` (device accumulators, float32 confidence to match the reference). Generations are **bit-identical to the pre-§A code** (stash A/B, 4/4). §A5 (collapse the 3× metadata rebuild) was **measured and dropped** — F3-lite phase timing showed metadata is only ~5% of wall (the rebuild is not the bottleneck); fragile shared-backend surgery for ~3%.

## Validation
- **α→∞ ≡ LowConfidence** on `experiments/dllm/focus_a100_smoke` (anchor; the 3/4 vs LowConfidence is a pre-existing split-vs-monolithic FP-path difference, identical on old and new code, not a regression).
- **Redundancy** Σ|S|/(B·bs) ramps 0.31→1.0 within a block; per-conc means 0.66/0.78/0.80 — real eviction.
- 9 test files green (7 CPU unit + 2 GPU micro-tests). GPU micro-tests de-risk the FlashInfer reduced-attention mechanism and the CUDA-graph capture on real hardware.

## Performance (experiment F1, `experiments/profiling/dllm/focus_vs_lowconf`)
LLaDA2.0-mini, 1×A100, TP=1, **eager**, HumanEval at conc {1,8,16}:
- **§A throughput vs LowConfidence: 0.91 / 0.80 / 0.77×** (was 0.90/0.84/0.69× pre-§A; §A is +9% absolute at conc-16, flat at 1/8). FOCUS does NOT beat LowConfidence in this eager, small-MoE regime.
- **Why:** F3-lite phase split (conc-8) = `s_fwd ~62%` (Phase S = 18 MoE layers, eager per-layer launch latency) · `select ~14%` (host Python loop) · `p_fwd ~12%` · metadata ~5% · commit ~2%. The eviction (FLOPs cut) is real; the wall-clock loss is host/launch overhead on a tiny model, not GPU compute.
- **Levers (reprioritized by the timing):** §C (Phase-S CUDA graph, the ~62%) and §B2 (selection Triton kernel, ~14%). §A5 dropped.

## §C — Phase-S CUDA graph: mechanism PROVEN, naive integration DEAD-ENDED
`algorithm/focus_graph.py` (bucketization/pad foundation, unit-tested) + `algorithm/focus_graph_runner.py` (`FocusPhaseSGraphRunner`, wired into `focus.py` behind `SGLANG_FOCUS_GRAPH=1`, **default OFF, eager fallback**).
- ✅ **Capture mechanism works:** FlashInfer paged-attn + the full MoE L2..L forward capture/replay correctly inside a CUDA graph; **α→∞ (single constant bucket) graph == eager bit-identical (4/4)**.
- 🔴 **Naive shape-keyed approach fails for real FOCUS (α=1.5, variable |S|), one root cause = variable shape:** (1) **churn** — |S| changes every step + context grows every block ⇒ shapes never stabilize ⇒ graphs re-capture constantly ⇒ capture cost ≫ eager ⇒ no throughput win; (2) **correctness** — multi-bucket padding path corrupts longer outputs (`!!!!`), not pool aliasing. The one naturally-fixed case (α→∞) is exactly the no-eviction case.
- **The real §C win (future milestone):** pad EVERY step to a fixed `(bs, qo_bucket, kv_bucket)` so one graph replays across many steps — isolated pad segment (pad never touches real attention), kv rounded to coarse buckets. Substantial; prerequisite regardless of model size.

## Not done / next
- **§B2** selection Triton kernel (port `focus_select_enforce_ragged`; my torch helper is the exact oracle) — the safer ~14% lever, not blocked by variable shape.
- **§C fixed-shape rework** — the real path to the ~62% lever.
- **§B1** importance Triton kernel; **DC+** behavioral wiring (decoded-in-block positions still reprocessed each step); persist FocusState across the worker boundary; SDAR port; F4 (compute-bound regime / longer ctx where L2..L dominate); Plan-B parallelism (TP/EP/DP).

## Config + launch
```yaml
# experiments/dllm/focus_a100_smoke/configs/focus.yaml
block_size: 32
threshold: 0.9        # commit confidence threshold
alpha: 1.5            # dynamic-budget expansion (α→∞ ⇒ no eviction ≡ LowConfidence)
maxpool_k: 3
min_retain: 1
importance_layers: [0, 1]
enable_delayed_cache: true
n_bar_init: 1.0
```
```bash
python -m sglang.launch_server --model-path <llada2> --dllm-algorithm Focus \
  --dllm-algorithm-config <focus.yaml> --tp-size 1 --mem-fraction-static 0.7 \
  --attention-backend flashinfer --disable-cuda-graph   # dLLM eager needs flashinfer pinned
# env: SGLANG_FOCUS_LOG_REDUNDANCY=1 / SGLANG_FOCUS_REDUNDANCY_CSV=<p> (redundancy),
#      SGLANG_FOCUS_PHASE_TIMING=1 (phase split), SGLANG_FOCUS_GRAPH=1 (Phase-S graph; default OFF)
```

## Files
```
algorithm/focus.py, focus_forward.py, focus_reduce.py, focus_utils.py   (algorithm + split metadata)
algorithm/focus_graph.py, focus_graph_runner.py                         (§C graph foundation + runner)
models/llada2.py                                                        (forward_focus_* split)
mixin/req.py                                                            (FocusState, DelayedCacheState)
test_focus_{utils,reduce,forward,selection_logic,state,graph,importance_axes}.py  (CPU, green)
test_focus_{reduced_attention,phase_s_graph}_gpu.py                     (A100 micro-tests, green)
experiments/dllm/focus_a100_smoke/                                      (α→∞ anchor harness)
experiments/profiling/dllm/focus_vs_lowconf/                           (F1 throughput experiment)
```

## References
- FOCUS paper `notes/26_FOCUS.pdf`; official impl `~/FOCUS_ORIGIN/` + `notes/code-walkthrough.md`
- Progress log `notes/focus_implementation_progress.md`; plan `notes/focus_graph_kernel_plan.md`; §C design `notes/focus_phase_s_graph_design.md`; parallelism `notes/focus_parallelism_plan.md`
