# FOCUS Implementation Summary

**Branch:** `feature/focus-implementation`  
**Date:** 2026-06-29  
**Status:** Phase A - Core Infrastructure (60% complete)

## What Has Been Implemented

### 1. Per-Request State Structures ✓
**Files:** `python/sglang/srt/dllm/mixin/req.py`

- **FocusState dataclass**
  - Tracks cumulative decoding statistics: `token_sum`, `total_steps`
  - Computes `avg_decoded_tokens` property for dynamic budgeting (Eq. 19)
  - `rightmost_processed` tracks furthest decoded position in current block
  - Persists cumulative statistics across blocks (reset only `rightmost_processed`)

- **DelayedCacheState dataclass**
  - Implements Neighbor-Aware Stability (DC+)
  - `uncached_positions`: bitmap tracking which positions need computation
  - `update_from_mask()`: marks position i cached when **both i and i+1 are decoded**
  - `get_processing_indices()`: returns indices requiring computation this step
  - `reset_for_new_block()`: resets all positions to uncached

- **ReqDllmMixin integration**
  - Initializes FOCUS states when `algorithm_name == "Focus"`
  - Automatically resets states on new block in `_init_fill_ids_for_dllm()`

**Tests:** `test_focus_state.py` - 13 test cases, all passing ✓

### 2. FOCUS Algorithm Skeleton ✓
**File:** `python/sglang/srt/dllm/algorithm/focus.py`

- **Focus class** extending `DllmAlgorithm`
  - Config parsing: `threshold` (0.9), `alpha` (1.5), `maxpool_k` (3), `min_retain` (1)
  - Enforces `enable_delayed_cache=True` as prerequisite
  - Basic denoising loop structure (mirrors LowConfidence)
  - TODO markers for: split forward, importance collection, selection, state updates

- **Algorithm registration**
  - Auto-discovered via `algo_name_to_cls` mechanism
  - Accessible via `--dllm-algorithm Focus`

### 3. Helper Functions (PyTorch Reference) ✓
**File:** `python/sglang/srt/dllm/algorithm/focus_utils.py`

- **`compute_importance_side_channel(q, k, seq_offsets, scaling, maxpool_k)`**
  - Computes intra-block attention scores `S_ij = q_i·k_j/√d`
  - Applies MaxPool1D(k=3) smoothing along key axis
  - Softmax over keys, sum over query and head → importance I_j (Eq. 2, 15)
  - CSR-format batching via `seq_offsets`

- **`compute_retention_budget(delta_I, avg_decoded, mask, seq_offsets, alpha, block_length)`**
  - Statistical thresholding: `threshold = mean(ΔI) + std(ΔI)` (1σ above mean)
  - N_σ = count of tokens ≥ threshold (Eq. 5)
  - Base budget from historical average: `ceil(α·N̄_decoded)`
  - Final budget: `K = min(B, max(base, N_σ))` (Eq. 4)

- **`select_and_enforce_constraints(delta_I, budgets, mask, seq_offsets, block_length, min_retain)`**
  - TopK selection from masked positions by importance delta
  - **AR-Context Preservation:** add predecessor i-1 for each selected i
  - **Placeholder Integrity:** retain all masked j < max(S)
  - **Minimum retention:** ensure |S| ≥ min_retain
  - Returns `retain_masks` and `retained_maps` per request

**Tests:** `test_focus_utils.py` - 9 test cases covering all functions, all passing ✓

## What Remains (Critical Path to Phase A Completion)

### 4. Split Model Forward (Highest Priority)
**Target files:** `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/sdar.py`

**Required changes:**
- Add `forward_focus_prefix(input_ids, positions, forward_batch, processing_mask)` to model classes
  - Run Layer 0 fully over processing positions (from delayed cache)
  - Run Layer 1 input_layernorm + QKV proj + RoPE over processing positions
  - **Critical:** Write full-block K,V to cache at Layer 1 BEFORE eviction
  - Collect importance scores I0, I1 via side-channel during L0/L1 attention
  - Return `(H1_full, I0, I1, q1, processing_global_indices)`

- Add `forward_focus_suffix(H1_reduced, positions_reduced, forward_batch_reduced, retained_to_block_map)`
  - Finish Layer 1 attention on reduced queries (vs full-block + context KV)
  - Run Layers 2..L on reduced token set
  - Return `(logits_retained, retained_to_block_map, layer_metas)`

**Challenges:**
- Need to modify attention layers to optionally collect importance scores
- Must handle ragged attention metadata rebuild after eviction
- Layer-specific logic (0 vs 1 vs 2+)

### 5. FocusRuntimeView and Batch Integration
**Target files:** `python/sglang/srt/manager/schedule_batch.py`, `python/sglang/srt/model_executor/forward_batch_info.py`

**Required:**
- Define `FocusRuntimeView` dataclass with pinned host buffers
- Build view in `get_model_worker_batch()` from per-request states
- Thread through `ForwardBatch.focus_view`
- Construct CSR-format sequence boundaries

### 6. Integrate Helpers into Algorithm
**Target file:** `python/sglang/srt/dllm/algorithm/focus.py`

**Flow per denoising step:**
1. Query delayed cache for `processing_indices` per request
2. Call `forward_focus_prefix` → get I0, I1
3. Compute `ΔI = I1 - I0` (host-side)
4. Call `compute_retention_budget` → get K per request
5. Call `select_and_enforce_constraints` → get retained masks/maps
6. Gather `H1_reduced = H1_full[retain_mask]`
7. Rebuild attention metadata for reduced queries
8. Call `forward_focus_suffix` → get logits for retained
9. Decode and commit (confidence > threshold | top-1)
10. Update `focus_state` (N̄_decoded, rightmost_processed)
11. Update `delayed_cache_state` (Neighbor-Aware caching)

### 7. Scheduler Feedback Loop
**Target files:** `python/sglang/srt/model_executor/model_agent.py`, scheduler

**Required:**
- Extract `processed_positions` from model metadata
- Call `scheduler.update_processed_positions(seq_ids, positions)`
- Update `delayed_cache_state.update_from_mask(dllm_mask, mask_id)`
- Update `focus_state` cumulative statistics

### 8. Integration Tests
**Create:**
- `test_focus_equivalence.py`: FOCUS with α→∞ should match LowConfidence
- `test_focus_e2e.py`: End-to-end generation quality on small model
- `test_delayed_cache.py`: Validate DC+ vs naive DC quality difference

## Testing Strategy

### Unit Tests (Complete ✓)
- State structures: 13 tests passing
- Helper functions: 9 tests passing
- Coverage: budgeting logic, selection constraints, Neighbor-Aware caching

### Integration Tests (Pending)
1. **Equivalence test:** α=∞, K=B should give identical output to LowConfidence
2. **Quality test:** Run on GSM8K sample, measure correctness
3. **DC+ validation:** Compare quality with/without Neighbor-Aware Stability

### System Tests (Future)
- Throughput benchmarking vs LowConfidence baseline
- Memory profiling
- Multi-GPU (deferred to later)

## Configuration

**YAML config example:**
```yaml
# config/focus.yaml
threshold: 0.9        # Confidence threshold for commit
alpha: 1.5            # Expansion factor for dynamic budgeting
maxpool_k: 3          # MaxPool kernel size for importance smoothing
min_retain: 1         # Minimum tokens to retain per step
enable_delayed_cache: true  # Must be true (DC+ prerequisite)
```

**Launch command:**
```bash
python -m sglang.launch_server \
  --model-path <path-to-llada2> \
  --port 30000 \
  --dllm-algorithm Focus \
  --dllm-algorithm-config config/focus.yaml \
  --dllm-block-size 32 \
  --attention-backend flashinfer \
  --disable-cuda-graph \
  --mem-fraction-static 0.7
```

## Known Limitations (Phase A)

1. **No split forward yet:** Currently uses standard monolithic forward
2. **No eviction yet:** Processes all tokens every step (like LowConfidence)
3. **No CUDA graph:** Eager execution only
4. **Single GPU only:** No TP/PP support
5. **No Triton kernels:** PyTorch reference implementations (Phase B task)

## Next Steps

**Immediate (to complete Phase A correctness):**
1. Implement split model forward (prefix/suffix) for LLaDA2MoeAttention
2. Add importance collection side-channel in attention layers
3. Integrate helper functions into Focus.run() denoising loop
4. Add scheduler feedback loop for state updates
5. Write equivalence and quality integration tests
6. Validate on single A100 with small model

**Phase B (performance optimization):**
- Replace PyTorch helpers with Triton kernels
- Fused importance computation
- Fused selection and state compaction
- Ragged paged attention kernel

**Phase C (production readiness):**
- Hybrid CUDA graph bucketization
- Multi-GPU support (TP/PP)
- Sparse KV cache fill
- Profiling and tuning

## Files Changed

```
notes/
  focus_implementation_progress.md     (tracking)
  focus_sglang_implementation_plan.md  (refined plan)

python/sglang/srt/dllm/
  mixin/req.py                         (+60 lines: FocusState, DelayedCacheState)
  algorithm/focus.py                   (+169 lines: Focus class skeleton)
  algorithm/focus_utils.py             (+240 lines: helper functions)
  test_focus_state.py                  (+156 lines: state tests)
  test_focus_utils.py                  (+166 lines: helper tests)
```

**Total new code:** ~791 lines  
**Tests:** 22 test cases, all passing ✓

## Architecture Decisions

1. **DC+ as prerequisite:** Following official implementation, not optional
2. **PyTorch first:** Correctness before Triton optimization
3. **CSR format:** Ragged batching via `seq_offsets` for variable-length sequences
4. **Pinned host buffers:** Avoid D2H sync for scheduling metadata
5. **Modular helpers:** Easy to swap PyTorch → Triton in Phase B

## References

- FOCUS paper: `notes/26_FOCUS.pdf` (ICML 2026)
- Official implementation: `~/FOCUS_ORIGIN/` + `notes/code-walkthrough.md`
- Implementation plan: `notes/focus_sglang_implementation_plan.md`
- Progress tracking: `notes/focus_implementation_progress.md`
