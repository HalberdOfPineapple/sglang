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
