# FOCUS Implementation Progress

Branch: `feature/focus-implementation`
Start Date: 2026-06-29
Target: Phase A correctness (eager, with DC+, single GPU)

> ⚠️ **CRITICAL (2026-06-29, post-review):** the "Phase A" below is a
> **logit-masking** realization that reproduces FOCUS's *decoding schedule* but
> runs every transformer layer on all B tokens — it saves **ZERO FLOPs** and is
> therefore NOT a real FOCUS implementation. The paper's compute win comes from
> physically evicting tokens after Layer 1 and running L1-attn..L on `|S| ≪ B`.
> The paper-exact reduced forward is specified in
> **`notes/focus_paper_exact_plan.md`** — that is the source of truth.
>
> **UPDATE (2026-06-30):** the host-side math is now paper-correct and tested
> (the N_σ-in-budget bug is FIXED; selection/compaction match the official
> kernels and are unit-pinned — plan §0.1). The only remaining gap to real FLOPs
> savings is the split forward, now specified via per-phase `seq_lens` +
> contiguous-prefix KV compaction (plan §8, R1 resolved) — paper-exact with **no**
> custom Triton kernel and no page_size change.

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
