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
- **C2 — Reduced-set (ragged) attention.** After eviction, queries = retained tokens (variable `|S_b|` per request), keys/values = full context + full block (already in cache). This is an extend with `q_len < kv_len` and non-contiguous intra-block queries. **Approach:** write full-block KV at Layer 1, then rebuild attention metadata for the retained queries and run the standard extend attention path. Phase A reuses FlashInfer extend with recomputed `cu_seqlens`/positions; Phase C adds a ragged kernel.
- **C3 — Model forward must be split** into eager prefix (L0 full + L1 QKproj + selection + gather) and suffix (L1 attn..L on `S`). The generic `model_runner.forward` runs a monolithic layer loop, so FOCUS needs a dedicated model entry `forward_focus_prefix` / `forward_focus_suffix`.
- **C4 — Dynamic shapes vs CUDA graph.** `|S|` varies per step. dLLM already disables piecewise CUDA graph. Phase A runs eager; Phase C adds FOCUS's hybrid bucketization (power-of-2 < 256, stride-256 ≥ 256; Eq. 21/22).
- **C5 — Per-request persistent state.** `N̄_decoded`, `rightmost_processed`, and `uncached_positions` persist across steps/blocks per request → store on the `Req` dLLM mixin.
- **C6 — Serial stepping.** FOCUS requires per-step state updates (its "disable multi-loop" constraint, Appendix E.3.4). SGLang's `disable_overlap_schedule=True` for dLLM already gives serial stepping — no extra work, just preserve it.

## 3. Phased Strategy (correctness first)
- **Phase A — Correct reference (eager, no freeze, no graph).** New `Focus` algorithm + split model forward + side-channel importance + budgeting/selection in vectorized PyTorch + reduced-set extend attention. Target: bit-for-bit acceptable generation quality matching paper Table 3/4 trends; throughput parity-or-better at large batch without kernels. This is the milestone that proves correctness.
- **Phase B — Intra-Block KV cache (DC+).** Add Neighbor-Aware delayed caching to skip recomputing frozen tokens' KV across steps. Validate against Table 6 (DC+ must not regress; full FOCUS should match/beat baseline).
- **Phase C — Performance kernels & graph.** Triton `focus_importance_ragged`, `focus_select_enforce_ragged`, `focus_compact_states`, ragged paged attention, sparse KV fill, and hybrid-bucketized CUDA-graph suffix. Target paper's 2.3–3.5× throughput.

## 4. Per-Component Modifications

### 4.1 New algorithm `Focus` — `python/sglang/srt/dllm/algorithm/focus.py` (new)
Mirror `LowConfidence` structure and registration (`Algorithm = Focus` at module end so the registry in `dllm/algorithm/__init__.py` picks it up). Responsibilities = host-side control flow of Algorithm 1:
- `__init__(config)`: read `alpha` (default 1.5), `threshold` (default 0.9, paper default `Conf=0.8`), `maxpool_k=3`, `min_retain=1`, `importance_layers=(0,1)` from `config.algorithm_config`.
- `run(model_runner, forward_batch)`:
  1. Fast path unchanged: if no `mask_id` in `input_ids`, single `model_runner.forward` to populate KV, return.
  2. Per-block `start_list` as today.
  3. Denoising loop (≤ `block_size` iters): instead of `model_runner.forward`, call a new `model_runner.forward_focus(forward_batch, focus_state)` that returns `(logits_for_retained, retained_index_map, can_run_cuda_graph)`. Commit decodable tokens (confidence > threshold, else top-1) **only among retained positions**, writing into `forward_batch.input_ids` in place — evicted positions stay `mask_id` and are revisited next step.
  4. Update per-request `N̄_decoded` cumulative mean with the per-step commit count (Eq. 17–19); update `rightmost_processed`.
  5. Final forward + reshape to variable-length `next_token_ids_list` exactly as `low_confidence.py:92`.
- Budgeting/selection helpers (`compute_targets`, `select_and_enforce`) live here in Phase A (pure PyTorch, vectorized over `[batch, B]`), to be swapped for Triton kernels in Phase C. Keep them as standalone functions so the kernel swap is local.

### 4.2 Config & flags — `dllm/config.py`, `server_args.py`
- No new CLI flag strictly required: `--dllm-algorithm Focus --dllm-algorithm-config focus.yaml` already routes through `DllmConfig.from_server_args` → `get_algorithm`. The YAML carries `alpha`, `threshold`, `block_size`, `maxpool_k`.
- Add to `DLLM_PARAMS` the importance-layer count assumption (needs `num_hidden_layers ≥ 2`); assert at init.
- Preserve the forced overrides in `server_args._handle_dllm_inference()` (flashinfer backend, `disable_overlap_schedule`, `disable_piecewise_cuda_graph`, `page_size=block_size`, `pp_size=1`). FOCUS adds no relaxation in Phase A; Phase C re-enables a FOCUS-specific CUDA-graph capture path (Section 4.8) guarded by an `enable_focus` flag on the cuda-graph runner.

### 4.3 Per-request state — `python/sglang/srt/dllm/mixin/req.py`
Add a `FocusState` carried on `Req` (init in `init_diffusion_llm`, reset on block completion in `init_next_round_input`/`_init_fill_ids_for_dllm`):
- `token_sum`, `total_steps` → `n_bar_decoded = token_sum/total_steps` (Eq. 19); **persists across blocks** for stable budgeting.
- `rightmost_processed` (Eq. 16), reset to `−1` per block.
- `uncached_positions: BoolTensor[block_size]` for DC+ (Phase B), reset to all-True per block.
The algorithm reads/writes these through the `ForwardBatch` (thread a per-request `FocusRuntimeView` into `forward_batch`, analogous to `dllm_block_offsets`). Pack as tensors in `get_model_worker_batch` (`schedule_batch.py:2609`) and surface on `ForwardBatch.init_new` (`forward_batch_info.py`).

### 4.4 Split model forward — `python/sglang/srt/models/llada2.py`
This is the core change. Add a focused forward path to `LLaDA2MoeModel`/`LLaDA2MoeModelLM` (mirror for SDAR models if targeting SDAR too):
- `forward_focus_prefix(input_ids, positions, forward_batch)`:
  - Embed + run **Layer 0 fully** (attn + MLP) over the full block; capture Layer-0 intra-block scores → `I0` (Section 4.5).
  - Run **Layer 1 input_layernorm + QKV projection + RoPE** over the full block; write full-block **K,V to the KV cache** (so retained queries can attend to all block keys); capture Layer-1 intra-block scores → `I1`.
  - Return `H1_full` (pre-attention hidden states for L1), `I0`, `I1`, and the L1 q tensor.
- Host (algorithm) computes `ΔI = I1 − I0`, budget `K`, set `S` (Section 4.6), then `H1_reduced = gather(H1_full, S)` with reduced positions.
- `forward_focus_suffix(H1_reduced, positions_reduced, forward_batch_reduced)`:
  - Finish **Layer 1 attention** (retained queries vs full-block + context KV) + Layer-1 MLP on `S`.
  - Run **Layers 2..L** on `S` (the standard layer loop, but over `|S|·batch` tokens).
  - Final norm + `LogitsProcessor(return_full_logits=True)` → logits for retained tokens only.
- Implementation note: keep the existing monolithic `forward` for the LowConfidence path untouched; gate on `forward_batch.forward_mode`/a `focus` flag so non-FOCUS dLLM and AR serving are unaffected (CLAUDE.md: isolate dLLM changes).

### 4.5 Importance side-channel — `LLaDA2MoeAttention` (`llada2.py:518`)
Add an opt-in branch that, when `forward_batch.focus_collect_scores` is set, computes the intra-block importance contribution from the **post-RoPE** `q,k` already in `forward`:
- For each request block of `B` tokens: `S_block = einsum('hid,hjd->hij', q_b, k_b) * scale` → `[H,B,B]`.
- `MaxPool1D(k=3)` along the key axis `j`, `Softmax` over `j`, sum over query `i` and head `h` → `I_j ∈ R^B` (Eq. 2/15). Aggregate across TP attention heads with an all-reduce over the attention-TP group (heads are sharded; importance must sum over all heads).
- Return `I` to the model forward without altering the normal `self.attn(...)` output. At Layer 1 the function is called in "QKproj-only" mode (skip the full `self.attn` for evicted tokens, but still fill KV). Cost is `O(B²Hd)` per request — negligible (Appendix E.5).
- Phase A computes this in PyTorch; Phase C replaces with `focus_importance_ragged` Triton kernel (fused MaxPool+softmax+column-sum with shared-memory atomics, Appendix E.2.1) to handle ragged per-request `B` and decoded-token freezing.

### 4.6 Budgeting & selection — in `focus.py` (Phase A), Triton later
Vectorized over `[batch, B]`:
- `N_σ = Σ_j 1(ΔI_j ≥ Std(ΔI_j over masked j))` per request (Eq. 5); `K = min(B, max(ceil(α·n_bar), N_σ))`, with `n_bar=1` on step 1 (Eq. 4).
- `S = topk(ΔI masked, K)` per request; restrict candidates to currently-masked positions only.
- **AR-Context Preservation:** `S ← S ∪ {i−1 : i∈S}`.
- **Placeholder Integrity:** `S ← S ∪ {j : j < max(S), j is masked}` (retain masked prefix up to the max selected index).
- **Min retention:** ensure `|S| ≥ 1`.
- Produce per-request `retained_index_map` (local block indices) used for the gather and to map retained logits back to absolute block positions on commit. Phase C: `focus_select_enforce_ragged` does this in one pass (Appendix E.2.2).

### 4.7 Reduced-set attention metadata — attention backend glue
Phase A (reuse FlashInfer extend):
- After full-block KV is written at L1, build a reduced `ForwardBatch`/attention metadata where, per request, the **extend queries are the `|S_b|` retained tokens** and the KV range covers context + full block. Set `extend_seq_lens=|S_b|`, recompute `cu_seqlens_q`, and use the retained tokens' **original absolute positions** for RoPE (already known from `dllm_block_offsets`). Re-init the attention backend's forward metadata for this reduced shape (call the backend's `init_forward_metadata` again with the reduced batch), then run layers `1_suffix..L`.
- `out_cache_loc` for the suffix must point retained queries at their already-allocated KV slots (no new KV writes for layers ≥1 beyond what L1 wrote for the full block; suffix layers write KV for retained tokens only — acceptable because evicted tokens are not decoded this step and will be recomputed next step in Phase A).
- Phase C: replace with `ragged_paged_attention_fwd` (indirect `tile_to_seq` mapping, Appendix E.2.4) + `fill_kv_cache_sparse` to avoid the metadata rebuild and support non-contiguous queries natively.

### 4.8 Intra-Block KV cache / DC+ (Phase B) — `dllm/mixin` + attention glue
- `DelayedCacheState.uncached_positions` governs which block positions are recomputed each step. Warmup: all uncached (full block processed). After a step, mark position `i` cached only when `i` and `i+1` are both decoded (Neighbor-Aware, Section 4.3/E.3.2). Block reset clears it.
- In `forward_focus_prefix`, skip Q/K/V recompute for cached positions (their KV is frozen in cache); only uncached positions participate in L0/L1 full compute and importance. This shrinks the "full block" cost too.
- Validate strictly against Table 6: naive DC regresses (e.g. GSM8K 89.2→84.9); DC+ restores it. Gate behind `enable_delayed_cache` in the YAML so Phase A correctness is unaffected.

### 4.9 CUDA-graph hybrid bucketization (Phase C) — `model_runner` cuda-graph runner
- Keep prefix (L0/L1 + selection + gather) **eager** (dynamic shapes); capture only the **suffix** (`forward_focus_suffix`) as graphs keyed on `N_retained` total tokens.
- Buckets: power-of-2 `{1,2,4,…,128}` for `N_retained<256`, else `ceil(N_retained/256)·256` (Eq. 21/22). At runtime pick the smallest graph `≥ N_retained` and pad. Adjust warmup capture (`focus_cap=(max_batches+1)//2`, Appendix E.4.3) to capture reduced batch sizes.
- Add `enable_focus` to the cuda-graph runner's input strategy; leave `disable_piecewise_cuda_graph` as-is for the prefix.

### 4.10 Scheduler & result write-back — mostly unchanged
- Batch construction (`get_new_batch_dllm`, `DLLM_EXTEND`) is reused verbatim; FOCUS lives below the worker hand-off (`tp_worker.py:435`). The modified `scheduler.py` / `topk.py` in the working tree (A100 sm80 topk fallback, see `[[a100-sm80-flashinfer-topk-fallback]]`) are unrelated and stay.
- `process_batch_result_dllm` already handles variable-length per-request commits, so FOCUS's per-step variable commit count needs no change there. Confirm `next_token_ids_list` shaping matches.
- Preserve serial stepping (no overlap) — required by FOCUS state dependencies (C6).

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
