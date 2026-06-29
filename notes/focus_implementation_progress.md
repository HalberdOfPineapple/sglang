# FOCUS Implementation Progress

Branch: `feature/focus-implementation`
Start Date: 2026-06-29
Target: Phase A correctness (eager, with DC+, single GPU)

## Implementation Checklist

### Phase A: Core Infrastructure
- [ ] Per-request state structures (FocusState, DelayedCacheState)
- [ ] FocusRuntimeView and pinned buffers
- [ ] Config and algorithm registration
- [ ] Split model forward (prefix/suffix)
- [ ] Importance side-channel (PyTorch)
- [ ] Budgeting and selection (PyTorch)
- [ ] State compaction and reduced attention
- [ ] Scheduler feedback loop
- [ ] DC+ Neighbor-Aware cache management

### Testing
- [ ] Unit tests for FocusState/DelayedCacheState
- [ ] Unit tests for budgeting/selection logic
- [ ] Unit tests for importance computation
- [ ] Integration test: FOCUS vs LowConfidence equivalence (α→∞)
- [ ] End-to-end generation quality test
- [ ] DC+ validation (with vs without)

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

### 2026-06-29: FOCUS algorithm skeleton (Milestone 2)
- **Created `Focus` algorithm class** (`python/sglang/srt/dllm/algorithm/focus.py`)
  - Extends `DllmAlgorithm` base class
  - Reads config: `threshold`, `alpha`, `maxpool_k`, `min_retain`, `enable_delayed_cache`
  - Enforces DC+ as prerequisite (raises error if disabled)
  - Implements basic denoising loop (mirrors LowConfidence structure)
  - TODO markers for split forward, importance, selection
- **Algorithm auto-registration**
  - Uses existing `algo_name_to_cls` discovery mechanism
  - Accessible via `--dllm-algorithm Focus`
