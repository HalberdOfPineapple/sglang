# FOCUS on SGLang — Implementation Plan

## 0. Source and Scope
This plan maps the FOCUS paper (`notes/26_FOCUS.pdf`, ICML 2026, KAUST/CUHK) onto SGLang's existing dLLM stack. FOCUS is a **training-free, intra-step token-eviction** system for block-diffusion LLMs (SDAR, LLaDA2.0). The reference implementation targets LMDeploy; here we re-target it to SGLang's `--dllm-algorithm` path, reusing the LLaDA2/SDAR models, the `DLLM_EXTEND` scheduling path, and `RadixAttention`. This is a **plan, not an implementation** — it lists per-component modifications, concrete file/function anchors, and a phased rollout. Companion reading: `notes/llada2_workflow_and_parallelism.md` (end-to-end dLLM call chain) and `notes/dllm_llada2_lowconfidence_launch_walkthrough.md`.

## 1. What FOCUS Does (algorithmic recap)
Block-diffusion decoding computes the full block of `B` query tokens through all `L` layers every denoising step, yet only ~10% of tokens are decodable per step (Fig. 2) — so ~90% of FLOPs are wasted, and because DLLM decoding is **compute-bound** (Table 2: FLOPs scale linearly with block query size `B`), batching gains plateau. FOCUS removes the waste by **evicting non-decodable query tokens after the first two layers** and running layers `2..L` only on a small retained set `S`. Core mechanism (Algorithm 1, Appendix C):
- **Importance** `I_j = Σ_{i,h} Softmax(MaxPool1D(S_ij^(h), k=3))` where `S_ij^(h) = q_i·k_j/√d` is the intra-block pre-softmax score, aggregated column-wise (Eq. 2). Computed per layer.
- **Importance delta** `ΔI_j = I_j^(Layer1) − I_j^(Layer0)` (Eq. 3) — a Common-Mode-Rejection that cancels Layer-0 positional priors. ΔI correlates strongly with decodability (Fig. 3/4); Layer 1 is the earliest layer where decodable tokens diverge.
- **Dynamic budget** `K = min(B, max(⌈α·N̄_decoded⌉, N_σ))` (Eq. 4), where `α>1` (default 1.5) is the only new hyperparameter, `N̄_decoded` is the per-request cumulative-mean decode yield (default 1 at step 1), and `N_σ = Σ_j 1(ΔI_j ≥ Std(ΔI))` counts tokens ≥1σ above the zero-mean delta distribution (Eq. 5).
- **Selection** `S = TopK(ΔI, K)`, then two structural constraints: **AR-Context Preservation** (force-retain predecessor `i−1` of each candidate — CPT backbones rely on `t_i ← t_{i-1}`), and **Placeholder Integrity** (retain all masked tokens with index `< max(S)` so evicted positions keep valid reference KV and correct relative offsets). Minimum `|S| ≥ 1`.
- **Reduced forward** Gather `H` to `H_reduced ∈ R^{|S|×h}`, run layers `1_suffix..L` only on `S`; evicted tokens stay masked this step but keep fixed KV. `|S| ≪ B` linearly cuts matmul FLOPs.
- **Intra-Block KV Cache** (Section 4.3) Neighbor-Aware Stability Criterion: freeze a decoded token `t_i`'s KV only once **both** `t_i` and right-neighbor `t_{i+1}` are decoded (Delayed Cache `DC+`), avoiding the quality drop of naive delayed caching (Table 6).
- KV for the **full block is written before eviction at Layer 1** (Appendix C), so retained queries attend to all block keys.

## 2. SGLang Mapping and Key Challenges
The existing dLLM path (`LowConfidence`) already gives us: `DLLM_EXTEND` batches with `[MASK]`-padded blocks, per-block absolute `positions` for RoPE (`forward_batch_info.py:538`), encoder-only `RadixAttention`, full-logits output, and a denoising loop that mutates `forward_batch.input_ids` in place (`dllm/algorithm/low_confidence.py:23`). FOCUS is **orthogonal to and composable with** LowConfidence's parallel commit: LowConfidence decides *which masked tokens to commit* from full-block logits; FOCUS decides *which tokens to keep computing* inside one forward. FOCUS replaces the per-step "full forward over all `B·batch` tokens" with a "focused forward" that yields logits only for the retained set.

Key challenges and chosen approach:
- **C1 — Attention scores are not exposed.** FlashInfer/FA `RadixAttention` returns only the context vector, not `S_ij`. **Approach:** compute a cheap **side-channel intra-block score** `S_block = q_block·k_blockᵀ/√d` restricted to the block (`[H,B,B]`, `B=32` → trivial `O(B²Hd)`, Appendix E.5 confirms negligible) directly from the post-RoPE `q,k` already materialized in `LLaDA2MoeAttention.forward`. No attention-kernel surgery in Phase A.
- **C2 — Reduced-set (ragged) attention.** After eviction, queries = retained tokens (variable `|S_b|` per request), keys/values = full context + full block (already in cache). This is an extend with `q_len < kv_len` and non-contiguous intra-block queries. **Approach:** write full-block KV at Layer 1 **BEFORE eviction** (critical timing), then rebuild attention metadata for the retained queries using CSR-format ragged tensors (`seq_offsets` for sequence boundaries) and run the standard extend attention path. Phase A reuses FlashInfer extend with recomputed `cu_seqlens`/positions; Phase C adds a ragged paged attention kernel.
- **C3 — Model forward must be split** into eager prefix (L0 full + L1 QKproj + selection + gather) and suffix (L1 attn..L on `S`). The generic `model_runner.forward` runs a monolithic layer loop, so FOCUS needs a dedicated model entry `forward_focus_prefix` / `forward_focus_suffix`.
- **C4 — Dynamic shapes vs CUDA graph.** `|S|` varies per step. dLLM already disables piecewise CUDA graph. Phase A runs eager; Phase C adds FOCUS's hybrid bucketization (power-of-2 < 256, stride-256 ≥ 256; Eq. 21/22).
- **C5 — Per-request persistent state.** `N̄_decoded` (cumulative mean), `rightmost_processed`, and `uncached_positions` persist across steps/blocks per request → store on the `Req` dLLM mixin. Use **pinned host memory** for scheduling metadata (`block_progress`, `avg_tokens`, `mask_seq_offsets`) to avoid D2H synchronization overhead.
- **C6 — Serial stepping.** FOCUS requires per-step state updates (its "disable multi-loop" constraint, Appendix E.3.4). SGLang's `disable_overlap_schedule=True` for dLLM already gives serial stepping — no extra work, just preserve it.
- **C7 — Delayed cache is prerequisite, not optional.** The official implementation treats Neighbor-Aware delayed cache (DC+) as a hard prerequisite for FOCUS correctness, not a Phase B optimization. Without it, KV cache instability degrades quality. Must be integrated from Phase A.

## 3. Phased Strategy (correctness first)
- **Phase A — Correct reference (eager, with DC+, no graph).** New `Focus` algorithm + split model forward + side-channel importance + budgeting/selection in vectorized PyTorch + reduced-set extend attention + **Neighbor-Aware delayed cache (DC+)**. DC+ is a **prerequisite**, not optional: without it, KV cache instability causes quality degradation (Table 6: naive DC drops GSM8K 89.2→84.9; DC+ restores it). Target: generation quality matching paper Table 3/4 trends; throughput parity-or-better at large batch without kernels. This is the milestone that proves correctness.
- **Phase B — Fused Triton kernels.** Replace PyTorch ops with fused Triton kernels: `focus_importance_ragged` (QK scoring + maxpool + softmax + delta), `focus_compute_targets` (dynamic budgeting), `focus_select_enforce_ragged` (selection + structural constraints), `focus_compact_states` (prefix sum + gather all state tensors). Target: reduce per-step overhead from ~1.5% to <1%.
- **Phase C — Ragged attention & CUDA graph.** Ragged paged attention kernel (tile-based, CSR-format sequence boundaries), sparse KV cache fill, and hybrid-bucketized CUDA-graph suffix (prefix stays eager). Target paper's 2.3–3.5× throughput.

## 4. Per-Component Modifications

### 4.1 New algorithm `Focus` — `python/sglang/srt/dllm/algorithm/focus.py` (new)
Mirror `LowConfidence` structure and registration (`Algorithm = Focus` at module end so the registry in `dllm/algorithm/__init__.py` picks it up). Responsibilities = host-side control flow of Algorithm 1:
- `__init__(config)`: read `alpha` (default 1.5), `threshold` (default 0.9, paper default `Conf=0.8`), `maxpool_k=3`, `min_retain=1`, `importance_layers=(0,1)`, `enable_delayed_cache=True` (hard requirement) from `config.algorithm_config`.
- `run(model_runner, forward_batch)`:
  1. Fast path unchanged: if no `mask_id` in `input_ids`, single `model_runner.forward` to populate KV, return.
  2. Per-block `start_list` as today.
  3. **Per-step loop**: query delayed cache state for `processing_indices` (which positions need computation this step — initially all uncached, shrinks as tokens are cached per DC+). Build `processing_mask` from these indices.
  4. Call `model_runner.forward_focus(forward_batch, focus_view)` with `processing_mask`, returns `(logits_for_retained, retained_to_block_map, layer_metas)`.
  5. **Decode and verify**: apply confidence threshold (>0.9) or top-1 to `logits_for_retained`, map back to block positions via `retained_to_block_map`, commit decodable tokens into `forward_batch.input_ids` (unmask), update DLLM mask state.
  6. **Feedback loop**: extract `processed_positions` from `layer_metas[DLLM_META_PROCESSED_TOKENS]`, call `scheduler.update_processed_positions(seq_ids, processed_positions)` to update delayed cache state (mark positions cached per DC+ neighbor-aware rule) and focus state (`N̄_decoded` cumulative mean, `rightmost_processed`).
  7. Final forward + reshape to variable-length `next_token_ids_list` exactly as `low_confidence.py:92`.
- Helper functions: `_build_focus_view(forward_batch, processing_mask, focus_states)` constructs the `FocusRuntimeView` with pinned host tensors for scheduling metadata.

### 4.2 Config & flags — `dllm/config.py`, `server_args.py`
- No new CLI flag strictly required: `--dllm-algorithm Focus --dllm-algorithm-config focus.yaml` already routes through `DllmConfig.from_server_args` → `get_algorithm`. The YAML carries `alpha`, `threshold`, `block_size`, `maxpool_k`.
- Add to `DLLM_PARAMS` the importance-layer count assumption (needs `num_hidden_layers ≥ 2`); assert at init.
- Preserve the forced overrides in `server_args._handle_dllm_inference()` (flashinfer backend, `disable_overlap_schedule`, `disable_piecewise_cuda_graph`, `page_size=block_size`, `pp_size=1`). FOCUS adds no relaxation in Phase A; Phase C re-enables a FOCUS-specific CUDA-graph capture path (Section 4.8) guarded by an `enable_focus` flag on the cuda-graph runner.

### 4.3 Per-request state — `python/sglang/srt/dllm/mixin/req.py`
Add dataclasses carried on `Req` (init in `init_diffusion_llm`, reset on block completion in `init_next_round_input`/`_init_fill_ids_for_dllm`):

#### 4.3.1 `FocusState`
Tracks cumulative decoding statistics and block progress per request:
```python
@dataclass
class FocusState:
    block_length: int
    token_sum: int = 0           # Cumulative decoded tokens across all steps
    total_steps: int = 0          # Total denoising steps taken
    rightmost_processed: int = -1 # Furthest decoded position in current block
    
    @property
    def avg_decoded_tokens(self) -> float:
        """Cumulative mean N̄_decoded (Eq. 19) for dynamic budgeting."""
        return self.token_sum / max(self.total_steps, 1)
```
**Persistence**: `token_sum` and `total_steps` persist **across blocks** for stable budgeting (model learns per-request decodability rate). `rightmost_processed` resets per block.

#### 4.3.2 `DelayedCacheState`
Manages Intra-Block KV Cache with Neighbor-Aware Stability (DC+):
```python
@dataclass
class DelayedCacheState:
    block_length: int
    uncached_positions: torch.Tensor  # BoolTensor[block_length], True = needs computation
    needs_warmup: bool = True         # First step processes all positions
    
    def get_processing_indices(self) -> torch.Tensor:
        """Returns indices of positions to compute this step."""
        return torch.where(self.uncached_positions)[0]
    
    def update_from_mask(self, dllm_mask: torch.Tensor, mask_id: int):
        """Apply Neighbor-Aware Stability: mark position i cached when both i and i+1 decoded."""
        unmasked = (dllm_mask != mask_id)
        # ready_mask[i] = unmasked[i] AND unmasked[i+1]
        ready_mask = unmasked[:-1] & unmasked[1:]
        ready_mask = torch.cat([ready_mask, unmasked[-1:]])  # Last position ready when decoded
        self.uncached_positions &= ~ready_mask
        self.needs_warmup = False
```
**Key mechanism**: CPT architectures have local dependency (t_{i+1} attends heavily to t_i), so freezing t_i's KV before t_{i+1} decodes injects noise. DC+ waits for both neighbors before caching. Reset to all-True per block.

#### 4.3.3 Pinned host memory pool
Scheduling metadata that changes every step but is needed on host should use **pinned host memory** to avoid D2H sync overhead:
```python
class FocusPinnedBuffers:
    """Pre-allocated pinned memory for zero-copy host→device transfers."""
    def __init__(self, max_batches: int, max_tokens: int):
        self.processing_q_lens = torch.zeros(max_batches, dtype=torch.int32, pin_memory=True)
        self.processing_indices = torch.zeros(max_tokens, dtype=torch.int32, pin_memory=True)
        self.focus_block_progress = torch.zeros(max_batches, dtype=torch.int32, pin_memory=True)
        self.focus_avg_tokens = torch.zeros(max_batches, dtype=torch.float32, pin_memory=True)
        self.focus_mask_seq_offsets = torch.zeros(max_batches + 1, dtype=torch.int32, pin_memory=True)
```
These are packed in `schedule_batch.py:get_model_worker_batch` and surfaced on `ForwardBatch` via `FocusRuntimeView` (Section 4.5).

### 4.4 Split model forward — `python/sglang/srt/models/llada2.py`
This is the core change. Add a focused forward path to `LLaDA2MoeModel`/`LLaDA2MoeModelLM` (mirror for SDAR models if targeting SDAR too):

#### 4.4.1 `forward_focus_prefix(input_ids, positions, forward_batch, processing_mask)`
Runs L0 full + L1 QKproj over **processing positions only** (from delayed cache):
- Embed tokens at `processing_indices` (not full block — DC+ already cached some positions).
- Run **Layer 0 fully** (attn + MLP) over processing positions; capture Layer-0 intra-block scores → `I0` via `_collect_importance_scores()`.
- Run **Layer 1 input_layernorm + QKV projection + RoPE** over processing positions.
- **CRITICAL**: Write full-block **K,V to the KV cache at Layer 1** for all processing positions (so retained queries can attend to all block keys after eviction). Use attention backend's `forward_only_fill_kv()` method.
- Capture Layer-1 intra-block scores → `I1`.
- Return `(H1_full, I0, I1, q1, processing_global_indices)` where `H1_full` is pre-attention hidden states for L1, `q1` is the query tensor, and `processing_global_indices` maps local indices to absolute block positions.

#### 4.4.2 Host-side selection (in `model_runner` or `algorithm`)
- Compute `ΔI = I1 − I0` (only for masked positions).
- **Statistical thresholding**: `threshold = mean(ΔI_masked) + std(ΔI_masked)` (1σ above mean, Eq. 5).
- **Dynamic budgeting**: `K = min(B, max(⌈α·N̄_decoded⌉, N_σ))` where `N_σ = sum(ΔI ≥ threshold)` (Eq. 4).
- **TopK selection**: `S_candidates = topk(ΔI_masked, K)`.
- **AR-Context Preservation**: `S ← S ∪ {i−1 : i∈S, i>0}` (force-retain predecessors).
- **Placeholder Integrity**: `S ← S ∪ {j : j < max(S), j is masked}` (retain masked prefix up to max index).
- **Min retention**: `|S| ≥ 1`.
- Produce `retain_mask[processing_positions]` and `retained_to_block_map` (local retained indices → absolute block positions).
- Gather `H1_reduced = H1_full[retain_mask]`, `positions_reduced = positions[retain_mask]`.

#### 4.4.3 `forward_focus_suffix(H1_reduced, positions_reduced, forward_batch_reduced, retained_to_block_map)`
Runs L1 attention..L over retained set only:
- Finish **Layer 1 attention** on `H1_reduced` (queries = retained, KV = full context + full block already in cache).
- Run **Layer 1 MLP** on retained hidden states.
- Run **Layers 2..L** (standard layer loop) over `|S|·batch` tokens.
- Final norm + `LogitsProcessor(return_full_logits=True)` → logits for retained tokens only, shape `[num_retained_total, vocab_size]`.
- Return `(logits_retained, retained_to_block_map, layer_metas)` where `layer_metas[DLLM_META_PROCESSED_TOKENS]` contains the `processing_indices` for feedback.

#### 4.4.4 Implementation notes
- Keep existing monolithic `forward` for LowConfidence/AR paths untouched.
- Gate on `forward_batch.forward_mode == "focus"` or a `focus_view is not None` check.
- Use CSR-format ragged tensors for batch: `seq_offsets[i]:seq_offsets[i+1]` delineates sequence `i`'s tokens.
- In Phase A, rebuild FlashInfer attention metadata after gather; Phase B replaces with ragged paged attention.

### 4.5 FocusRuntimeView — `python/sglang/srt/manager/schedule_batch.py` + `forward_batch_info.py`
Package all FOCUS runtime state into a single view that flows through the forward pass:

#### 4.5.1 Dataclass definition (in `forward_batch_info.py` or `model_inputs.py`)
```python
@dataclass
class FocusRuntimeView:
    """Device-side tensors for FOCUS execution."""
    processing_mask_global_indices: torch.Tensor  # [total_processing] int32, global block positions
    processing_mask_evictable: bool               # Can eviction occur this step?
    block_progress: torch.Tensor                  # [batch] int32, pinned host memory
    avg_tokens: torch.Tensor                      # [batch] float32, pinned host memory (N̄_decoded)
    mask_seq_offsets: torch.Tensor                # [batch+1] int32, CSR format sequence boundaries
    importance_l0: Optional[torch.Tensor] = None  # [batch, B] float32, filled at Layer 0
    importance_l1: Optional[torch.Tensor] = None  # [batch, B] float32, filled at Layer 1
```

#### 4.5.2 Construction (in `schedule_batch.py:get_model_worker_batch`)
Build `FocusRuntimeView` from per-request states:
```python
# Query delayed cache for processing indices per request
processing_indices_per_req = [req.delayed_cache_state.get_processing_indices() for req in batch]
total_processing = sum(len(idx) for idx in processing_indices_per_req)

# Pack into pinned buffers (allocated once, reused across steps)
pinned = focus_pinned_buffers  # Singleton pool
offset = 0
for i, (req, indices) in enumerate(zip(batch, processing_indices_per_req)):
    n = len(indices)
    pinned.processing_q_lens[i] = n
    pinned.processing_indices[offset:offset+n] = req.dllm_block_offsets[indices]  # Map to global
    pinned.focus_block_progress[i] = req.focus_state.rightmost_processed
    pinned.focus_avg_tokens[i] = req.focus_state.avg_decoded_tokens
    offset += n
    pinned.focus_mask_seq_offsets[i+1] = offset

focus_view = FocusRuntimeView(
    processing_mask_global_indices=pinned.processing_indices[:total_processing].cuda(non_blocking=True),
    processing_mask_evictable=(total_processing > 0),
    block_progress=pinned.focus_block_progress[:len(batch)],  # Stay on pinned host
    avg_tokens=pinned.focus_avg_tokens[:len(batch)],          # Stay on pinned host
    mask_seq_offsets=pinned.focus_mask_seq_offsets[:len(batch)+1].cuda(non_blocking=True),
)
```

#### 4.5.3 Threading through `ForwardBatch`
Add `focus_view: Optional[FocusRuntimeView]` field to `ForwardBatch`. Populate in `ForwardBatch.init_new()` when `forward_mode == "dllm_focus"`. Model layers read/write `forward_batch.focus_view.importance_l0/l1` during prefix.

### 4.6 Importance side-channel — `LLaDA2MoeAttention` (`llada2.py:518`)
Add an opt-in branch that, when `layer_idx in [0, 1]` and `forward_batch.focus_view.processing_mask_evictable`, computes the intra-block importance contribution from the **post-RoPE** `q,k` already in `forward`:

#### 4.6.1 Phase A: PyTorch side-channel
```python
def _compute_focus_importance(self, layer_idx: int, q: Tensor, k: Tensor, 
                               focus_view: FocusRuntimeView) -> Tensor:
    """Compute importance scores I_j = Σ_{i,h} Softmax(MaxPool1D(S_ij^h, k=3)) (Eq. 2)."""
    # q, k: [num_processing_tokens, num_heads, head_dim] (post-RoPE)
    # Build per-request intra-block score matrices
    batch_importance = []
    for b in range(len(focus_view.mask_seq_offsets) - 1):
        start = focus_view.mask_seq_offsets[b]
        end = focus_view.mask_seq_offsets[b + 1]
        q_b = q[start:end]  # [B_processing, H, d]
        k_b = k[start:end]
        
        # S_block = q·kᵀ/√d → [H, B, B]
        S = torch.einsum('bhd,Bhd->hbB', q_b, k_b) * self.scaling
        
        # MaxPool1D(k=3) along key axis, then softmax
        S_pooled = F.max_pool1d(S.flatten(0, 1).unsqueeze(1), kernel_size=3, 
                                 stride=1, padding=1).squeeze(1).view_as(S)
        S_soft = torch.softmax(S_pooled, dim=-1)  # Softmax over keys (last dim)
        
        # Sum over query and head → I_j ∈ R^B
        I = S_soft.sum(dim=(0, 1))  # [B]
        batch_importance.append(I)
    
    importance = torch.cat(batch_importance)  # [total_processing]
    
    # TP all-reduce: attention heads are sharded, must sum across TP group
    if self.tp_size > 1:
        torch.distributed.all_reduce(importance, group=self.tp_group)
    
    return importance

# In SDARAttention.forward() / LLaDA2MoeAttention.forward():
if self.layer_idx == 0 and focus_view and focus_view.processing_mask_evictable:
    focus_view.importance_l0 = self._compute_focus_importance(0, q, k, focus_view)
elif self.layer_idx == 1 and focus_view and focus_view.processing_mask_evictable:
    focus_view.importance_l1 = self._compute_focus_importance(1, q, k, focus_view)
```

#### 4.6.2 Phase B: Triton `focus_importance_ragged` kernel
Fused MaxPool+softmax+column-sum with shared-memory atomics (Appendix E.2.1), handles ragged per-request `B` and CSR-format sequence boundaries. Cost `O(B²Hd)` per request — negligible (Appendix E.5).

### 4.7 Budgeting & selection — in `focus.py` (Phase A), Triton later
Host-side selection logic (PyTorch in Phase A, Triton `focus_compute_targets` + `focus_select_enforce_ragged` in Phase B):

#### 4.7.1 Dynamic budgeting (Eq. 4, 5)
```python
def compute_retention_budget(delta_I: Tensor, avg_decoded: Tensor, alpha: float, 
                             block_length: int, mask: Tensor) -> Tensor:
    """Compute K = min(B, max(⌈α·N̄_decoded⌉, N_σ)) per request."""
    batch_size = len(avg_decoded)
    budgets = []
    
    for b in range(batch_size):
        delta_I_b = delta_I[b][mask[b]]  # Only masked positions
        
        # Statistical threshold: μ + σ (1σ above mean)
        threshold = delta_I_b.mean() + delta_I_b.std()
        N_sigma = (delta_I_b >= threshold).sum().item()
        
        # Base budget from historical average
        base_budget = math.ceil(alpha * avg_decoded[b].item())
        
        # Final budget
        K = min(block_length, max(base_budget, N_sigma))
        budgets.append(K)
    
    return torch.tensor(budgets, dtype=torch.int32)
```

#### 4.7.2 Selection with structural constraints
```python
def select_and_enforce(delta_I: Tensor, budgets: Tensor, mask: Tensor, 
                       block_length: int) -> Tuple[Tensor, List[Tensor]]:
    """TopK + AR-Context + Placeholder Integrity + min retention."""
    batch_size = delta_I.size(0)
    retain_masks = []
    retained_maps = []
    
    for b in range(batch_size):
        K = budgets[b].item()
        delta_I_b = delta_I[b]
        mask_b = mask[b]  # True = masked (candidate for selection)
        
        # TopK among masked positions
        masked_indices = torch.where(mask_b)[0]
        if len(masked_indices) == 0:
            # No masked tokens, retain nothing (or handle edge case)
            retain_masks.append(torch.zeros(block_length, dtype=torch.bool))
            retained_maps.append(torch.empty(0, dtype=torch.int64))
            continue
        
        delta_I_masked = delta_I_b[masked_indices]
        topk_in_masked = torch.topk(delta_I_masked, min(K, len(masked_indices))).indices
        S = masked_indices[topk_in_masked]
        
        # AR-Context Preservation: add i-1 for each i in S
        predecessors = S - 1
        predecessors = predecessors[predecessors >= 0]  # Valid indices only
        S = torch.unique(torch.cat([S, predecessors]))
        
        # Placeholder Integrity: retain all masked j < max(S)
        if len(S) > 0:
            max_S = S.max().item()
            placeholder_indices = torch.arange(max_S + 1, device=delta_I.device)
            placeholder_indices = placeholder_indices[mask_b[:max_S + 1]]
            S = torch.unique(torch.cat([S, placeholder_indices]))
        
        # Min retention
        if len(S) == 0:
            S = masked_indices[:1]  # Retain at least one masked token
        
        # Build retain_mask and map
        retain_mask = torch.zeros(block_length, dtype=torch.bool, device=delta_I.device)
        retain_mask[S] = True
        retain_masks.append(retain_mask)
        retained_maps.append(S)
    
    return torch.stack(retain_masks), retained_maps
```

#### 4.7.3 Phase B: Triton kernels
- `focus_compute_targets`: Parallel per-request budgeting with warp-level reductions for μ, σ.
- `focus_select_enforce_ragged`: Fused TopK + constraint enforcement in one pass (Appendix E.2.2).

### 4.8 State compaction & reduced-set attention — model/attention glue

#### 4.8.1 Compacting hidden states (Phase A: PyTorch gather, Phase B: Triton `focus_compact_states`)
After selection, gather all model states to dense retained-only format:
```python
def compact_states_for_suffix(H1: Tensor, q1: Tensor, residual: Tensor,
                               retain_masks: Tensor, retained_maps: List[Tensor]) -> Tuple:
    """Gather hidden states, Q, residual for retained tokens."""
    # Flatten batch dimension for gather
    total_retained = sum(len(m) for m in retained_maps)
    H1_reduced = torch.empty((total_retained, H1.size(-1)), dtype=H1.dtype, device=H1.device)
    q1_reduced = torch.empty((total_retained, *q1.shape[1:]), dtype=q1.dtype, device=q1.device)
    residual_reduced = torch.empty_like(H1_reduced)
    
    offset = 0
    for b, (retain_mask, retained_map) in enumerate(zip(retain_masks, retained_maps)):
        n = len(retained_map)
        # Gather from per-request slice
        start_b = sum(retain_masks[:b].sum() for b in range(b))  # Offset into flattened batch
        H1_reduced[offset:offset+n] = H1[start_b:start_b+len(retain_mask)][retain_mask]
        q1_reduced[offset:offset+n] = q1[start_b:start_b+len(retain_mask)][retain_mask]
        residual_reduced[offset:offset+n] = residual[start_b:start_b+len(retain_mask)][retain_mask]
        offset += n
    
    return H1_reduced, q1_reduced, residual_reduced
```
**Phase B**: Triton `focus_compact_states` fuses prefix sum over `retain_mask` + gather for all tensors in one pass, reducing kernel launch overhead.

#### 4.8.2 Reduced-set attention metadata (Phase A: rebuild FlashInfer, Phase C: ragged paged attention)
After full-block KV is written at L1, build attention metadata for the reduced query set:

**Phase A approach** (reuse FlashInfer extend):
```python
def rebuild_attention_metadata_for_retained(forward_batch: ForwardBatch, 
                                             retained_maps: List[Tensor]) -> ForwardBatch:
    """Build new ForwardBatch with reduced queries, full KV."""
    # Compute new seq_lens (queries only)
    extend_seq_lens_reduced = torch.tensor([len(m) for m in retained_maps], dtype=torch.int32)
    
    # Recompute cu_seqlens_q (cumulative sum)
    cu_seqlens_q_reduced = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        extend_seq_lens_reduced.cumsum(0)
    ])
    
    # Retained tokens' original positions (for RoPE in suffix attention)
    positions_reduced = []
    for b, retained_map in enumerate(retained_maps):
        block_offset = forward_batch.dllm_block_offsets[b]
        positions_reduced.append(block_offset + retained_map)
    positions_reduced = torch.cat(positions_reduced)
    
    # KV metadata unchanged: context + full block already in cache
    # Only query-side metadata changes
    forward_batch_reduced = forward_batch.copy()
    forward_batch_reduced.extend_seq_lens = extend_seq_lens_reduced
    forward_batch_reduced.cu_seqlens_q = cu_seqlens_q_reduced
    forward_batch_reduced.positions = positions_reduced
    
    # Re-init attention backend metadata
    forward_batch_reduced.attention_backend.init_forward_metadata(forward_batch_reduced)
    
    return forward_batch_reduced
```
**Phase C**: Replace with `ragged_paged_attention_fwd` (tile-based, CSR-format sequence boundaries via `tile_to_seq` indirect mapping, Appendix E.2.4) + `fill_kv_cache_sparse` to avoid metadata rebuild and support non-contiguous queries natively.

### 4.9 CUDA-graph hybrid bucketization (Phase C) — `model_runner` cuda-graph runner
- Keep prefix (L0/L1 + selection + gather) **eager** (dynamic shapes); capture only the **suffix** (`forward_focus_suffix`) as graphs keyed on `N_retained` total tokens.
- Buckets: power-of-2 `{1,2,4,…,128}` for `N_retained<256`, else `ceil(N_retained/256)·256` (Eq. 21/22). At runtime pick the smallest graph `≥ N_retained` and pad. Adjust warmup capture (`focus_cap=(max_batches+1)//2`, Appendix E.4.3) to capture reduced batch sizes.
- Add `enable_focus` to the cuda-graph runner's input strategy; leave `disable_piecewise_cuda_graph` as-is for the prefix.

### 4.10 Scheduler feedback loop — `model_agent.py` + `schedule_batch.py`
After each forward pass, propagate processed positions back to scheduler to update state:

#### 4.10.1 Model agent extracts metadata (`model_agent.py`)
```python
# In ModelAgent.process_batch_result():
if forward_mode == "dllm_focus":
    # Extract which positions were actually processed this step
    processed_positions = model_metas.get(DLLM_META_PROCESSED_TOKENS)  # List[Tensor] per request
    seq_ids = [req.request_id for req in batch]
    
    # Feedback to scheduler
    scheduler.update_processed_positions(seq_ids, processed_positions)
```

#### 4.10.2 Scheduler updates request state (`schedule_batch.py` or `scheduler_dllm.py`)
```python
def update_processed_positions(self, seq_ids: List[str], processed_positions: List[Tensor]):
    """Update DelayedCacheState and FocusState after a denoising step."""
    for seq_id, positions in zip(seq_ids, processed_positions):
        req = self.get_request(seq_id)
        
        # Update delayed cache with Neighbor-Aware Stability
        req.delayed_cache_state.update_from_mask(req.dllm_mask, self.mask_id)
        
        # Update focus state
        if len(positions) > 0:
            req.focus_state.rightmost_processed = max(
                req.focus_state.rightmost_processed, 
                positions.max().item()
            )
        
        # Update cumulative statistics (after commit, not just processing)
        num_decoded_this_step = (req.dllm_mask != self.mask_id).sum().item() - req.focus_state.token_sum
        if num_decoded_this_step > 0:
            req.focus_state.token_sum += num_decoded_this_step
            req.focus_state.total_steps += 1
```
This feedback loop is critical for DC+ (marks positions as cached) and dynamic budgeting (updates `N̄_decoded`).

## 5. Data-Flow Summary (one focused step)
`forward_focus` per denoising step: embed full block → **L0 full** (attn+MLP) + collect `I0` → **L1 QKproj** full + write full-block KV + collect `I1` → host: `ΔI=I1−I0`, `K` (Eq. 4), `S` (TopK + AR-context + placeholder) → gather `H1→H1_reduced` → rebuild reduced attention metadata → **L1 attn..L on S** → logits for `S` → commit decodable (conf>threshold | top-1) into `input_ids`, update `N̄_decoded`, `rightmost_processed`, `uncached_positions` → next step sees fewer masks and (Phase B) fewer uncached positions.

## 6. Validation Plan
- **Correctness (Phase A gate):** run `Focus` vs `LowConfidence` on LLaDA2.0-mini (4×A100 launch per `[[llada2-launch-config-a100]]`) on a few GSM8K/HumanEval prompts; outputs must be coherent and quality must track paper Table 3 (`Top > Random > Bottom`) and Table 4 (FOCUS ≥ baseline across `Conf∈{0.9,0.8,0.7}`, `α∈{1.2,1.5,1.8}`).
- **Redundancy metric:** log `N_processed/N_decoded` per block (Table 5); expect drop from ~15–20 toward ~3–4.
- **Throughput:** sweep batch `{32,64,128,256}` on ShareGPT/MATH; expect the LowConfidence baseline to plateau and Focus to scale (paper Fig. 6/7). Record per-denoising-iteration latency, not just tokens/s (note in `notes/`, store traces under CephFS per AGENTS.md).
- **Ablations:** Phase B — DC vs DC+ (Table 6); Phase C — eager vs graph suffix, bucket-count vs warmup time.
- Add a CI smoke test for the `Focus` registration + a tiny forward-equivalence test (with `α` huge so `K=B` ⇒ FOCUS must exactly equal LowConfidence) — this is the cleanest correctness anchor.

## 7. Risks & Open Questions
- **R1 (highest):** rebuilding attention metadata mid-forward for the reduced set (C2) may be costly or fight FlashInfer's planned metadata. Mitigation: the `α→∞` equivalence test isolates this; if metadata rebuild is too slow in Phase A, fall back to computing suffix on the full block but masking evicted logits (correct, no speedup) to validate selection logic first, then optimize.
- **R2:** importance requires summing attention heads across attention-TP shards — an extra small all-reduce per importance layer inside the denoising loop (paid every step). Quantify; should be tiny (`[B]`-sized) but appears in the hot loop.
- **R3:** post-RoPE vs pre-RoPE scores for `S_ij` — paper's `q_i·k_j/√d` is the actual attention score (post-RoPE). Use post-RoPE `q,k`; verify the ΔI/decodability correlation holds on our checkpoints (Fig. 4 replication on one batch).
- **R4:** Placeholder Integrity can inflate `|S|` toward `B` early in a block (max selected index large) — confirm the FLOPs win still materializes; this is expected and self-corrects as the block fills.
- **R5:** SDAR support needs the same split forward in the SDAR model file; LLaDA2.0-mini (MoE, 1.4B active) shows smaller relative gains (paper §5.3) — set throughput expectations accordingly.
- **Open:** whether to compute importance with the Phase-A side-channel forever (simple, ~1% overhead claimed) or fuse into the attention kernel; default to side-channel until profiling says otherwise.
