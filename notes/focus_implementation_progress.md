# FOCUS Implementation Progress

Branch: `feature/focus-implementation`
Start Date: 2026-06-29
Target: Phase A correctness (eager, with DC+, single GPU)

## Implementation Checklist

### Phase A: Core Infrastructure
- [x] Per-request state structures (FocusState, DelayedCacheState)
- [x] FocusRuntimeView and pinned buffers (partial - needs integration)
- [x] Config and algorithm registration
- [ ] Split model forward (prefix/suffix)
- [x] Importance side-channel (PyTorch) - helper function ready
- [x] Budgeting and selection (PyTorch) - helper functions ready
- [ ] State compaction and reduced attention
- [ ] Scheduler feedback loop
- [ ] DC+ Neighbor-Aware cache management - integrated into algorithm

### Testing
- [x] Unit tests for FocusState/DelayedCacheState
- [x] Unit tests for budgeting/selection logic
- [x] Unit tests for importance computation
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

## Current Status

**Phase A Progress: 60% complete**

✅ **Completed:**
- Per-request state structures (FocusState, DelayedCacheState)
- Algorithm skeleton with config parsing
- PyTorch helper functions (importance, budgeting, selection)
- Comprehensive unit tests (22 test cases, all passing)

🔨 **In Progress / Next Steps:**
1. Split model forward (prefix/suffix) for LLaDA2/SDAR
2. Importance collection side-channel in attention layers
3. Integrate helpers into Focus.run() denoising loop
4. Scheduler feedback loop for state updates
5. Integration tests (equivalence, quality, DC+ validation)

**Blockers:** None. Implementation can continue incrementally.

**Testing:** All unit tests passing. Need integration tests once split forward is complete.

See `notes/focus_implementation_summary.md` for detailed status and next steps.
