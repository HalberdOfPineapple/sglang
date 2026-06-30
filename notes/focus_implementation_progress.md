# FOCUS Implementation Progress

Branch: `feature/focus-implementation`
Start Date: 2026-06-29
Target: Phase A correctness (eager, with DC+, single GPU)

> ✅ **RESOLVED (2026-06-30, session 2):** the paper-exact reduced (token-evicting)
> forward is now IMPLEMENTED and validated end-to-end on a single A100 with
> LLaDA2.0-mini. `Focus.run` no longer does logit-masking — each denoising step
> runs the 3-phase split (P: L0 full + L1 QKV+fill; A1: L1 attn on |S| vs full-block
> KV; S: L2..L on |S|), so L1-attn..L execute on `|S| ≪ B` tokens. Measured
> `Σ|S|/(B·bs)` ramps 0.31→1.0 across a block (real FLOPs savings, matches the
> paper's redundancy curve). α→∞ anchor matches LowConfidence; α=1.5 stays coherent.
> Earlier sections (logit-masking "Phase A") are kept for history but are SUPERSEDED.

## Implementation Checklist

### Phase A: Correctness via logit-masking eviction (COMPLETE)
- [x] Per-request state structures (FocusState, DelayedCacheState)
- [x] FocusRuntimeView threaded on ForwardBatch.focus_view
- [x] Config and algorithm registration (`--dllm-algorithm Focus`)
- [x] Importance side-channel wired into LLaDA2MoeAttention (layers 0,1)
- [x] Budgeting and selection integrated into Focus.run()
- [x] Logit-masking eviction (Phase A realization, no reduced forward yet)
- [x] Per-request cumulative decode stats driving the budget K
- [x] End-to-end run on single A100 (LLaDA2.0-mini)
- [x] α→∞ ⇒ bit-for-bit identical to LowConfidence (validated)

### Testing (COMPLETE for Phase A)
- [x] Unit tests for FocusState/DelayedCacheState (13)
- [x] Unit tests for budgeting/selection logic (9)
- [x] Unit tests for importance computation
- [x] Selection-logic tests incl. α→∞ retain-all anchor (5)
- [x] Integration test: FOCUS @ α→∞ vs LowConfidence (4/4 prompts MATCH)
- [x] End-to-end generation quality test (FOCUS @ α=1.5 coherent)

### Deferred to Phase C (reduced forward)
- [ ] Split model forward (prefix/suffix) — needed for real FLOPs savings
- [ ] State compaction + reduced/ragged attention
- [ ] Scheduler feedback loop persisting FocusState across worker boundary
- [ ] DC+ behavioral effect (no-op under Phase-A full-forward; see note below)

> **DC+ scope clarification.** `DelayedCacheState` (Neighbor-Aware Stability)
> is implemented and unit-tested, but is a **no-op in Phase A**: the logit-masking
> realization runs a *full* forward over the whole block every step, so no KV is
> ever frozen mid-block and there is nothing to destabilize. DC+ only changes
> behavior once the **reduced forward** (Phase C) skips recomputing cached
> positions. Kept wired + tested so Phase C only needs to consume it.

## Changes Log

### 2026-06-29: Initial setup
- Created branch `feature/focus-implementation`
- Created progress tracking document

### 2026-06-29: Per-request state structures (Milestone 1)
- **Added `FocusState` dataclass** (`python/sglang/srt/dllm/mixin/req.py`)
  - Tracks `token_sum`, `total_steps`, `rightmost_processed`
  - Computes `avg_decoded_tokens` for dynamic budgeting (Eq. 19)
  - Persists cumulative stats across blocks
- **Added `DelayedCacheState` dataclass** (`python/sglang/srt/dllm/mixin/req.py`)
  - Implements Neighbor-Aware Stability (DC+)
  - Tracks `uncached_positions` bitmap
  - `update_from_mask()`: marks position i cached when both i and i+1 decoded
  - `get_processing_indices()`: returns positions needing computation
- **Updated `ReqDllmMixin.init_diffusion_llm()`**
  - Initializes FOCUS states when algorithm is "Focus"
  - Resets states on new block in `_init_fill_ids_for_dllm()`
- **Created unit tests** (`python/sglang/srt/dllm/test_focus_state.py`)
  - 13 test cases covering FocusState and DelayedCacheState
  - Validates Neighbor-Aware Stability logic
  - All tests pass ✓

### 2026-06-29: Helper functions and tests (Milestone 3)
- **Created FOCUS helper functions** (`python/sglang/srt/dllm/algorithm/focus_utils.py`)
  - `compute_importance_side_channel()`: Q·K scoring + MaxPool + Softmax (Eq. 2, 15)
  - `compute_retention_budget()`: Dynamic K with statistical thresholding (Eq. 4-5)
  - `select_and_enforce_constraints()`: TopK + AR-Context + Placeholder Integrity
- **Created unit tests** (`python/sglang/srt/dllm/test_focus_utils.py`)
  - 9 test cases for importance, budgeting, selection
  - Tests constraints: AR-Context Preservation, Placeholder Integrity, min retention
  - All tests pass ✓
- **Created implementation summary** (`notes/focus_implementation_summary.md`)
  - Comprehensive documentation of completed work
  - Detailed roadmap for remaining Phase A tasks
  - Architecture decisions and testing strategy

### 2026-06-29: End-to-end integration + single-A100 validation (Milestone 4)
- **Model integration** (`python/sglang/srt/models/llada2.py`)
  - Store `layer_id` on `LLaDA2MoeAttention`
  - `_collect_focus_importance()`: post-RoPE q,k → intra-block importance at
    layers 0/1 when `forward_batch.focus_view` set; GQA kv-head broadcast handled
- **Runtime view** (`python/sglang/srt/dllm/algorithm/focus_utils.py`)
  - `FocusRuntimeView` dataclass: per-layer importance collect + ΔI = I1 − I0
  - New `ForwardBatch.focus_view` field (`forward_batch_info.py`)
- **Algorithm rewrite** (`python/sglang/srt/dllm/algorithm/focus.py`)
  - Per-step: build focus_view → full forward (collect I0,I1) → ΔI → budget K →
    select_and_enforce → **logit-masking eviction** (suppress non-retained
    commits) → update per-request cumulative decode stats
  - Fixed `DllmConfig.algorithm` attribute name in `req.py`
- **Selection-logic tests** (`python/sglang/srt/dllm/test_focus_selection_logic.py`)
  - α→∞ ⇒ K=B ⇒ retain-all (LowConfidence equivalence anchor)
  - small-α eviction, AR-Context, Placeholder Integrity, ΔI — all pass ✓
- **Single-A100 smoke harness** (`experiments/dllm/focus_a100_smoke/`)
  - `run_smoke.sh` + `drive_smoke.py` + configs (focus / focus_alpha_inf / low_conf)
  - **FOCUS @ α→∞ == LowConfidence: 4/4 prompts bit-for-bit MATCH** ✓
  - **FOCUS @ α=1.5 (real eviction, ~13/32 retained): coherent + correct** ✓
- **Fix:** single-masked-token `std()` undefined → treat as N_σ=1

## Current Status

**Phase A: COMPLETE and validated on hardware.**

✅ **Completed & validated:**
- Per-request state structures (FocusState, DelayedCacheState) — unit tested
- Importance side-channel wired into LLaDA2 attention (layers 0,1)
- Budgeting + selection (TopK, AR-Context, Placeholder Integrity) integrated
- Logit-masking eviction realization (correct; equals LowConfidence at α→∞)
- End-to-end on single A100 80GB (LLaDA2.0-mini, TP=1, mem-frac 0.7)
- 27 unit/logic tests + 2 hardware integration checks, all green

🔨 **Next (Phase C — real FLOPs savings):**
1. Split model forward (prefix L0/L1 + suffix L1..L on retained set)
2. State compaction + reduced/ragged attention metadata
3. Scheduler feedback loop: persist FocusState across the worker boundary
   (thread through ModelWorkerBatch — currently algorithm only sees ForwardBatch)
4. DC+ becomes behaviorally active once reduced forward skips cached positions
5. Triton kernels (Phase B) + CUDA-graph suffix bucketization

**Key finding:** the algorithm receives only `forward_batch` (from a serialized
`ModelWorkerBatch`), not the scheduler `Req` objects, so persistent per-request
state needs to be threaded through `ModelWorkerBatch`. Phase A sidesteps this by
tracking cumulative stats locally within a single `run()` call (correct for
budgeting within a block-batch; cross-call persistence is a Phase C item).

See `notes/focus_implementation_summary.md` for the full roadmap.

### 2026-06-30 (session 3): Plan-A §A host-path de-sync LANDED (behavior-preserving; modest perf)
Implemented all of `focus_graph_kernel_plan.md` §A (de-sync the per-step host path, no kernels),
reducing the O(bs) D2H syncs/step to O(1). Each item validated; generations bit-identical.
- **A1 `_select_retained` batched** (`focus.py`): the per-request `mask_lengths` `.item()` loop →
  single `mask.view(bs,B).sum(1)` (uniform block_size processing set). Returns a stacked
  `[bs,B]` retain mask instead of a list.
- **A2 on-device index build** (`focus_reduce.build_retained_index_from_mask`,
  `focus_forward.build_phase_s_out_cache_loc`): `keep_index = retain_mask.flatten().nonzero()`,
  `new_lens = retain_mask.sum(1)`; out_cache_loc via `arange(B) < |S_b|` prefix-gather — both
  replace Python `.item()` loops. Loop versions kept as test oracles.
- **A3 single-D2H `make_focus_phase_batch`** (`focus_forward.py`): one `new_lens.cpu()` per step,
  shared by both reduced phases; all `*_cpu` FlashInfer-plan lists built by host arithmetic on
  `base_fb.seq_lens_cpu` (already host), device fields by device arithmetic — removed the two
  `.cpu()/.tolist()` round-trips per phase.
- **A4 vectorized `_commit_step`** (`focus.py`): per-request commit loop → one ragged pass —
  `keep_index`-scatter the |S|-flat argmax/softmax into dense `[bs,B]` confidence/predicted grids,
  threshold-or-top1 via `conf == row_max` (sync-free), one masked scatter into `input_ids`.
  `token_sum`/`total_steps` are now device int64 accumulators; `avg_decoded` computed on-device.
  Confidence pinned to **float32** to match the reference's default-dtype `−inf` buffer (avoids a
  bf16-vs-fp32 flip on borderline commits).
- **Tests**: `build_retained_index_from_mask` vs loop oracle + `new_lens_cpu`≡fallback added; all 5
  FOCUS test files green. **Anchor on HW**: stashed-vs-new A/B on `focus_a100_smoke` →
  NEW(§A) `focus_alpha_inf` == OLD `focus_alpha_inf` **4/4 bit-identical** (§A is behavior-
  preserving). The `alpha_inf==low_confidence` anchor is 3/4 on *both* old and new code (prompt 1
  differs by one word) — a **pre-existing** FP-path difference between the split forward and the
  monolithic forward, NOT a §A regression. α=1.5 coherent.
- **Perf (F1 re-run, same harness/model/HW; baseline 0.90/0.84/0.69×)**: speedup
  **0.91 / 0.80 / 0.77×** at conc 1/8/16; FOCUS absolute tok/s **60 / 233 / 290** vs baseline
  61 / 231 / 267 — i.e. **conc-16 +9%** (where the removed O(bs) loops had the most iterations),
  **flat at conc 1/8** (LowConf side moved ±5% run-to-run, so the ratios are noisy). Redundancy
  unchanged (0.662/0.801/0.802), confirming identical eviction.
  **Honest read:** §A does NOT flip FOCUS ahead of LowConfidence on this small MoE / eager regime,
  and does NOT flip conc-1 (the plan's guess) — conc-1 is bound by the **3× `init_forward_metadata`
  rebuild + eager launch latency**, not by O(bs) syncs (bs=1 ⇒ those loops were already O(1)). The
  remaining host cost is dominated by (i) the still-present Python **selection loop** inside
  `select_and_enforce_constraints` (§B2 Triton target — kept as oracle), (ii) the 3× metadata
  rebuild (§A5/§C), (iii) eager per-layer launch (§C). §A is necessary groundwork for §C (a graph
  needs a sync-light host path) but insufficient alone. Matches the plan's regime caveat: the
  decisive win needs §B2 + §C (and a compute-bound model/ctx, F4).
- **Not done / next**: §A5 (collapse A1+S metadata to one build), §B1/B2 kernels
  (`focus_importance_ragged`/`focus_select_enforce_ragged` — official kernels read this session,
  my torch helpers are exact oracles), §C Phase-S CUDA-graph capture.

### 2026-06-30 (session 3c): §C Phase-S graph — design + bucketization foundation LANDED
Chosen direction: §C (Phase-S CUDA-graph capture, the ~62% lever). It's a large ragged feature, so
staged like the split-forward (pure-tensor foundation + tests first, then GPU micro-test, then capture).
- **Design note** `notes/focus_phase_s_graph_design.md`: capture `forward_focus_rest_and_logits`
  (L2..L+norm+lm_head) keyed on **rounded total-|S|** (bucket ladder = pow2≤256 then stride-256, the
  official rule). Pad Σ|S|→bucket; real tokens `[0:Σ|S|)`, pad as a **trailing synthetic ragged
  segment** attending scratch KV; **pad KV writes → reserved scratch slot** (the top correctness
  invariant — padded compute must never corrupt a real block slot). Rewrite Phase-S FlashInfer metadata
  at replay (`seq_lens=context+|S|`, `prefix=context`, `out_cache_loc=block-prefix`), graph-break at the
  host selection on a side stream. Found the backend already has `is_dllm_extend` graph capture/replay
  branches (`flashinfer_backend.py:682,761`) but for the *full-block* regime — Phase S needs a reduced
  regime added. The existing `cuda_graph_runner.py` dLLM path (`:621`) captures only the *monolithic*
  forward (which FOCUS bypasses), so Phase S needs a bespoke capture.
- **Foundation** `algorithm/focus_graph.py` (pure tensor, no CUDA/model dep): `phase_s_token_bucket`
  (+`build_capture_token_buckets`), `build_phase_s_graph_layout` (ragged qo/kv segment lens with the
  trailing pad segment; Σqo==bucket), `pad_phase_s_out_cache_loc` (pad→scratch, KV-write safety),
  `pad_phase_s_tokens`. Unit-tested `test_focus_graph.py` (6 tests, green) incl. ladder values, the
  no-pad (Σ|S|==bucket) identity anchor, and the scratch-pad invariant.
- **§C-microtest ✅ DONE** (`test_focus_phase_s_graph_gpu.py`, A100): captured a CUDA graph around the
  FlashInfer Phase-S paged-prefill (`use_cuda_graph=True`, plan-outside/run-inside), replayed with a
  smaller real |S| padded to bucket (pad queries→scratch slot 0); real rows match the eager oracle
  (err 4.3e-3) and the graph **reuses** with new q content (err 4.6e-3). **Keystone §C risk retired:
  FlashInfer-in-graph + the pad/scratch-KV invariant both validated on hardware.**
- **§C-capture runner LANDED** (`algorithm/focus_graph_runner.py`, wired into `focus.py` behind
  `SGLANG_FOCUS_GRAPH=1`, default OFF, eager fallback on any failure): `FocusPhaseSGraphRunner` does
  **lazy** per-(bs,bucket) capture of `forward_focus_rest_and_logits` using the live padded `fb_s`
  (so the capturing run itself yields the correct step result); injects a dedicated
  `BatchPrefillWithPagedKVCacheWrapper(use_cuda_graph=True)` into `attn_backend.forward_metadata` for
  the Phase-S attention, plans the padded ragged layout (bs real + 1 scratch pad segment), copies real
  inputs into static buffers, replays, slices `[:Σ|S|]`. `_focus_reduced_forward` now returns
  `full_logits` directly (graph or eager).
- **Validation in progress (graph-ON smoke, iterating — eager fallback kept serving correct at every
  failure, 4/4 prompts):** bugs found & fixed: (1) `alloc(1)` on the paged dLLM allocator → 0 pages;
  reserving a page tripped SGLang's **pool memory-leak detector** → switched to **no allocation**: pad
  KV writes go to a per-step **non-retained block slot** (always exists when pad_len>0, overwritten by
  the block's final forward, never read by Phase S). (2) `attn_backend.req_to_token` doesn't exist →
  use `model_runner.req_to_token_pool.req_to_token`. (3) head-count/dtype attrs (`num_qo_heads`,
  `num_kv_heads`, `head_dim`, `data_type`) live on `attn_backend.indices_updater_prefill`, not the
  backend; `max_context_len` from `model_config.context_len`. Also widened the runner's try/except to
  cover the whole `run()` body (plan + buffer copies + capture) so a mid-step failure restores backend
  metadata and falls back to eager instead of crashing. Each failure was caught and fell back to eager
  (serving correct throughout). (4) **FlashInfer cuda-graph wrapper locks batch size (segment count) at
  construction** → a single wrapper can't serve varying bs. Rewrote to **one wrapper per bs** (lazy,
  sliced to bs segments, reused across buckets) AND folded the pad tokens into the **last real
  request's segment** (Option B) instead of a separate pad segment — keeps segment count == bs (no
  empty-segment edge case at pad_len=0), pad rows attend the last request's KV (non-causal, logits
  discarded), pad KV writes → a non-retained block slot.
- **✅ §C Phase-S GRAPH CAPTURE WORKS + α→∞ graph==eager BIT-IDENTICAL (4/4):** graph-ON
  `focus_alpha_inf` captured `(bs=1,bucket=32)` (no pad, |S|=B) and produced output **4/4 bit-identical
  to eager α→∞**. The MoE L2..L forward + FlashInfer paged attn capture & replay cleanly inside a
  CUDA graph.
- **Padding-path iteration (α=1.5, |S|<B ⇒ pad>0):** bucket `(1,16)` captured (sub-B padding works),
  but the FlashInfer cuda-graph wrapper **locks the total qo-row count at its first plan** (not just
  batch size) → a single per-bs wrapper can't serve multiple buckets (priming at max qo broke the real
  smaller qo). Rewrote to **one wrapper per (bs,bucket)** (qo fixed at bucket) + **re-capture on kv
  growth** (kv capacity is also first-plan-locked, but context accumulates across blocks). With that,
  all buckets `(1,4)`,`(1,8)`,`(1,16)`,`(1,32)` capture cleanly (no errors).

### 2026-06-30 (session 3d): §C honest status — mechanism PROVEN, but variable-|S| breaks the throughput premise
Two findings from running the real (α=1.5) padding path on hardware:
- **(perf, the killer) Re-capture churn.** FOCUS's retained count |S| changes EVERY denoising step
  (and context grows every block), so the per-step Phase-S shape (bucket + kv) is almost never stable.
  A graph keyed on exact shape therefore **re-captures constantly** — e.g. bucket `(1,32)` was captured
  8+ times in one short run as |S| ramped 17→32 within a block. Capture cost ≫ a single eager step, so
  with near-zero replay reuse the graph path would be SLOWER than eager. This is the fundamental
  CUDA-graph-vs-dynamic-shape tension: graphs amortize capture over many fixed-shape replays, but
  FOCUS Phase-S is intrinsically variable-shape. The ONE naturally-fixed case is α→∞ (|S|=B every step)
  — which is exactly the *no-eviction* case with no FOCUS benefit.
- **(correctness) Padding path corruption.** graph-ON α=1.5 produced `!!!!` garbage in the longer
  outputs (prompts 1,2; prompt 3 "2,3,5,7,11" was clean), while α→∞ (single constant bucket) was 4/4
  bit-identical. A `.clone()` of the output out of the graph pool did NOT fix it (reverted) ⇒ NOT
  output aliasing; it's a genuine bug in the variable-bucket padding capture/replay (only manifests
  with pad_len>0 + multiple buckets/re-captures). Not chased further — see the decision below.
- **DECISION (stop the naive path).** The naive shape-keyed graph is BOTH churn-limited (no throughput
  win) AND has a padding correctness bug, and both stem from the same root: FOCUS Phase-S is
  intrinsically variable-shape. Chasing the bug on the naive path is low value because even a correct
  naive path wouldn't speed FOCUS up. The proven, kept results are the milestones; the throughput win
  requires the fixed-shape dual-bucketing rework (below), which is the real §C task. `SGLANG_FOCUS_GRAPH`
  stays OFF; eager FOCUS (§A) + normal serving verified unaffected (defaults OFF, all tests green).
- **Fixed-shape rework (the actual path to a §C win), for a future session:** pad EVERY step to a
  fixed `(bs, qo_bucket, kv_bucket)` so one graph replays across many steps — qo padded into a
  dedicated **pad segment** (revive Option A; the now-fixed `bs+?` segment count makes it viable) whose
  queries AND keys are isolated (pad attends only pad/scratch KV, never real, so real logits are
  untouched), and kv padded up to `kv_bucket` per request. Round context growth to coarse kv buckets so
  re-capture is rare. This is the design that turns the proven capture mechanism into actual replay
  reuse. Substantial; scope as its own milestone.
- **Implication / design pivot for a real throughput win:** the graph must replay a FIXED shape across
  steps, which means padding BOTH qo AND kv to coarse buckets every step, with the pad-kv isolated so
  it can't corrupt real attention (the rejected "separate pad segment", Option A — which the
  fixed-`bs+1` segment count actually makes viable now). Then the same (bs, qo_bucket, kv_bucket) graph
  replays across many steps. That is a substantial rework, not a tweak.
- **What is solid and kept:** the §C foundation (`focus_graph.py` + tests), the design note, the
  micro-test (FlashInfer-in-graph + scratch-KV), and **α→∞ graph==eager bit-identical** — the capture
  mechanism itself is proven. `SGLANG_FOCUS_GRAPH` stays OFF by default; the working eager FOCUS path
  (§A, 0.91/0.80/0.77×) and normal serving are untouched.
- **Next:** confirm α→∞ graph==eager + α=1.5 coherent (exercises the pad/scratch path), then F1/F3
  re-measure (s_fwd device time ∝ 1−redundancy). §C4–C5 (prefix/A1 graph) after. Remaining risk is
  model-in-the-loop MoE-forward capture, not the attention mechanism (de-risked by the micro-test).

### 2026-06-30 (session 3b): F3-lite phase-time attribution → REPRIORITIZE (skip §A5, §C is the lever)
Added env-gated `SGLANG_FOCUS_PHASE_TIMING=1` (`focus.py`: `_phase` ctx-mgr, syncs at phase
boundaries, prints per-block `[focus-timing]`; no-op when unset, kept for §C validation/F3). Ran
FOCUS conc-8 on LLaDA2.0-mini. **Per-phase share of wall (bs=8, representative):**
`s_fwd ~62%` (Phase S = L2..L, 18 MoE layers on |S|) · `select ~14%` (host Python selection+compact
loop) · `p_fwd ~12%` (full-block prefix) · `a1_fwd ~4%` · `commit ~2%` · **`p+a1+s_meta ~5%`** (the
3× `init_forward_metadata` rebuild). Caveat: boundary syncs serialize, so absolute ms are inflated
and `s_fwd`'s share is somewhat overstated (exposed launch latency that would otherwise overlap) —
but the ordering is unambiguous: **s_fwd ≫ select > p_fwd ≫ metadata.**
**Consequences (revises the plan's cost table):**
- **§A5 is low-value — DROP it.** The 3× metadata rebuild is only ~5% of wall (not the dominant cost
  the plan assumed); §A5 collapses it via fragile surgery in the *shared* `flashinfer_backend.py` for
  a best-case ~3% — poor ROI. Marked done-by-decision (skipped), not implemented.
- **§C (Phase-S CUDA-graph capture) is confirmed THE lever** — ~62% of wall is L2..L eager per-layer
  launch latency on a tiny MoE; a graph collapses ~18×(attn+MoE) launches into one replay.
- **§B2 (selection Triton kernel) is the clear #2** — ~14% pure host (the `select_and_enforce_constraints`
  Python per-request loop, still present after §A; my torch helper is its exact oracle).
- New order: **§C1–C3 (Phase-S graph) → §B2 (selection kernel) → §C4–C5 (prefix/A1 graph)**; §A5 skipped.

### 2026-06-30 (session 2d): next-step plans written
- `notes/focus_graph_kernel_plan.md` (Plan A): de-sync host path → port Triton kernels
  (focus_importance/select/compact) → CUDA-graph capture of Phase S (then prefix/A1), to turn
  the F1 FLOPs cut into wall-clock speedup. Phased, each stage measured on the F1 harness.
- `notes/focus_parallelism_plan.md` (Plan B): TP/EP/DP compatibility. Core invariant = every rank
  derives the identical |S| (all-reduce the head-sharded importance over the attention-TP group +
  deterministic selection); TP/EP mostly free once that holds; DP-attention deferred. Shares the
  |S|-bucketization with Plan A.

### 2026-06-30 (session 2c): F1 throughput experiment — correct but host-bound (eager)
- New experiment `experiments/profiling/dllm/focus_vs_lowconf/` (run/parse/plot + README report;
  data mirror `/cephfs/shared/wxli/sglang-dllm/profiling/dllm/focus_vs_lowconf/`). FOCUS vs
  LowConfidence on LLaDA2.0-mini, 1×A100, TP=1, eager, HumanEval at conc {1,8,16}.
- **Result (honest negative): FOCUS is 0.90/0.84/0.69× LowConfidence throughput** (61/231/267 vs
  68/275/388 tok/s), gap widening with batch — despite genuinely evicting tokens (per-step
  Σ|S|/(B·bs) mean 0.662/0.784/0.803, i.e. 20–34% fewer tokens in L1-attn..L). Cause = eager
  per-step host overhead (3× FlashInfer `init_forward_metadata` rebuild + Python per-request
  selection/commit `.item()`/`.cpu()` syncs) >> the small per-token GPU FLOPs saved on a tiny MoE.
  The FLOPs cut is real; the wall-clock win needs CUDA-graph capture of Phase S + host-sync
  removal + a compute-bound regime (bigger model / longer ctx). Redundancy is the number to carry.
- Added robust redundancy logging: `SGLANG_FOCUS_REDUNDANCY_CSV=<path>` (flushed per step) +
  `SGLANG_FOCUS_LOG_REDUNDANCY=1` (print). Both no-ops when unset.

### 2026-06-30 (session 2b): Paper-exact reduced forward LANDED + validated on hardware
- **Model split** (`models/llada2.py`): `LLaDA2MoeAttention.forward_qkv_rope` (QKV+QK-norm+RoPE
  on the full block, collects importance, no attn/write), `.write_kv` (full-block KV fill via
  `set_kv_buffer`), `.forward_attn` (attn core + dense on a possibly-reduced q, `k=v=None`
  reads from cache); `LLaDA2MoeBlock.forward_focus_prefix_attn` (L1: prepare_attn + qkv_rope +
  fill) / `.forward_focus_suffix` (L1 attn on |S| + MLP); `LLaDA2MoeModel.forward_focus_prefix`
  (embed→L0 full→L1 prefix) / `.forward_focus_l1_suffix` / `.forward_focus_rest` (L2..L+norm);
  LM delegators incl. `.forward_focus_rest_and_logits`. The monolithic `forward` is untouched
  (normal serving unaffected).
- **`Focus.run` rewritten** to the 3-phase reduced forward: Phase P (full-block prefix) →
  host select+compact (`build_retained_index`/`index_select`) → Phase A1 (`make_focus_phase_batch`
  "A1", `init_forward_metadata`, L1 attn read-only on |S| vs full-block KV) → Phase S
  (`make_focus_phase_batch` "S" + `build_phase_s_out_cache_loc`, L2..L on |S|, KV→block prefix)
  → commit only the confident retained-masked positions. Forces `attn_backend.use_paged=True`
  (reduced phases read K/V from cache). `SGLANG_FOCUS_LOG_REDUNDANCY=1` prints Σ|S|/(B·bs).
- **Validated on 1×A100, LLaDA2.0-mini** via `experiments/dllm/focus_a100_smoke`:
  focus_alpha_inf (|S|=B) == low_confidence on all 4 prompts (anchor ✓); focus α=1.5 coherent
  with measured Σ|S|/(B·bs) ramping 0.31→1.0 within a block (real eviction / FLOPs savings ✓).
- **Key correctness facts:** dLLM block attention is non-causal (ENCODER_ONLY) so |S| reduced
  queries attend the full block independently; retained L≥2 KV written to the block's contiguous
  prefix so a single `req_to_token[req,0:context+|S|]` slice reads context+retained (no custom
  kernel). Per-phase metadata rebuilt by re-calling `init_forward_metadata` (eager).
- **Not yet done:** DC+ behavioral wiring (decoded-in-block positions are still reprocessed each
  step — eviction targets only non-retained masked, which is where the win is); persisting
  FocusState across the worker boundary (budget still tracked per-`run`); CUDA-graph capture of
  Phase S; SDAR port. None block correctness.

### 2026-06-30 (session 2): Per-phase metadata builder + GPU-validated reduced-attention keystone
- **Built the split-forward metadata builder** (`algorithm/focus_forward.py`, plan §8
  step 4): `compute_focus_phase_lens` (per-phase `seq_lens`/`extend_prefix_lens`/
  `extend_seq_lens` for Phase A1 q=|S| kv=context+B, and Phase S q=|S| kv=context+|S|),
  `build_phase_s_out_cache_loc` (contiguous block-prefix KV slots), `make_focus_phase_batch`
  (shallow-copy ForwardBatch applier, clears focus_view in the suffix). Pure tensor math,
  no model/CUDA dep ⇒ unit-testable. Anchored against the ORIGINAL dLLM-extend metadata
  (`seq_lens=context+B`, `extend_prefix_lens=context`, q=B) — verified in schedule_batch.py.
- **Unit test** `test_focus_forward.py` (5 tests, green): Phase A1/S lens math, the α→∞
  identity (|S|=B ⇒ both reduced phases reproduce the full-block metadata), out_cache_loc
  block-prefix gather, field stamping (base ForwardBatch untouched by shallow copy).
- **GPU micro-test** `test_focus_reduced_attention_gpu.py` (green on A100) — de-risks R4/R1
  on real hardware via the exact FlashInfer paged-prefill call SGLang's dLLM extend makes
  (page_size=1, causal=False, token-granular kv_indices), vs a torch dense-attention oracle:
  (a) full-block forward == oracle; (b) **Phase A1 q=|S|<kv reproduces the retained rows
  of the full-block forward** (FlashInfer accepts q_len<kv_len, non-causal rows independent);
  (c) **Phase S context+|S| contiguous-prefix reads exactly context+retained**. Needs
  `LD_PRELOAD=.../envs/sglang/lib/libstdc++.so.6` for the JIT .so (see memory note).
- **Status:** the split-forward keystone (per-phase metadata + the FlashInfer reduced-attn
  mechanism) is now built and hardware-validated. NOT yet done: the model-side split
  (`forward_focus_prefix` L0-full+L1-QKV+fill / `forward_focus_suffix` L1-attn..L on |S|,
  splitting `LLaDA2MoeAttention` into qkv_rope_fill + attention_from_q), the
  `model_runner.forward_focus` driver, and the `Focus.run` rewire — the remaining
  model-in-the-loop work, to be validated via the α→∞ smoke-harness anchor.

### 2026-06-30: Host-side correctness + compaction + de-risked split-forward plan
- **Fixed budget/selection** to match official kernels (Task 1): `compute_focus_targets`
  (no N_σ in budget), `compute_should_evict`, `select_and_enforce_constraints`
  (top-target OR N_σ mean+std expansion, AR-context adjacency, placeholder
  progress-gate). Rewrote `test_focus_utils.py` + `test_focus_selection_logic.py`
  (adds N_σ threshold-OR-topk pin). `Focus.run` uses new API.
- **Built state compaction** (Task 2): `focus_reduce.py` (build_retained_index,
  focus_compact_states, cu_seqlens_from_lens) + `test_focus_reduce.py` vs oracle.
- **De-risked the split forward**: R1 resolved — SGLang FlashInfer dLLM extend uses
  token-granular kv_indices (req_to_token contiguous slice), so paper-exact needs
  NO custom Triton kernel / NO page_size change. Mechanism = per-phase seq_lens +
  contiguous-prefix KV compaction (plan §8). KV-fill-only = direct set_kv_buffer
  (Task 3 needs no new method).
- All 5 FOCUS test files green. Remaining: the split forward itself (Task 4/5),
  a large end-to-end change requiring model-in-the-loop debugging — NOT started.
