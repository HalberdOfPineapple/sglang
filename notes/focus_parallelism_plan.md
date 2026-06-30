# FOCUS Plan B — Parallelism-Compatible Reduced Forward (TP / EP / DP)

**Goal:** make FOCUS run under the parallelisms `LowConfidence` already supports (TP, EP/MoE, DP-attention), single-node first (match the D2 config: 4×A100, TP4/EP4), without correctness loss or collective deadlock. 

LowConfidence is parallelism-agnostic *for free* because it only calls `model_runner.forward(forward_batch)` — the model layers handle TP all-reduce / EP all-to-all / DP gather internally. **FOCUS breaks this in three ways that this plan must fix.**



## Why FOCUS isn't parallel today (three breakages)
1. **Sharded importance.** `LLaDA2MoeAttention._collect_focus_importance` sums Softmax(MaxPool(q·k)) over **local** heads only; under TP the attention heads are sharded (`num_heads = total // attn_tp_size`), so each rank holds a *partial* importance. 
   - **ΔI differs per rank ⇒ selection differs per rank.**
2. **Data-dependent control that must be bit-identical across ranks.** FOCUS evicts a *content-dependent* token set |S|. 
   - The collectives in the reduced forward (TP all-reduce in `dense`/MLP, EP all-to-all in MoE, DP all-gather) are **lockstep** — if any rank picks a different |S|, token count, or order, the collectives desync ⇒ **NCCL hang / wrong output**. 
   - LowConfidence has no such per-step branch.
3. **The driver bypasses `model_runner.forward`.** `Focus.run` calls model methods (`forward_focus_prefix/_l1_suffix/_rest`) + `attn_backend.init_forward_metadata` **directly**, skipping the DP-attention / `global_num_tokens` / `dp_padding` / TBO setup that `model_runner.forward` does (model_runner.py:3281, the `global_num_tokens_*` / `dp_padding_mode` plumbing). 
   - So DP-attention metadata is never built for the reduced phases.



1. 

## The cardinal invariant
> **Every rank must derive the identical eviction decision** — same |S|, same retained block indices, same request-major token order — from identical inputs, every step. Achieve it by (a) all-reducing importance so ΔI is bit-identical on all ranks (NCCL all-reduce delivers the *same* output buffer to every rank), and (b) a **deterministic** selection kernel. Then no further sync is needed: `input_ids` are already replicated across TP/EP ranks for dLLM, so identical ΔI ⇒ identical selection ⇒ lockstep collectives.

## §A — Importance under TP (the one required new collective)
- After Phase P collects `I0`, `I1` (each `[B·bs]`, summed over **local** heads per rank), **all-reduce both across the attention-TP group** before selection: `tensor_model_parallel_all_reduce` (communication_op.py:18) or the attention-TP group's `all_reduce` (parallel_state.py:559) — use the *attention*-TP group (`get_attention_tp_group`) since heads are sharded there, not the full TP/EP world. This is the "single small all-reduce" the current code comment promises but never does.
- **Cost:** `bs·block_size·2` floats (e.g. bs16·32·2 = 1 K floats) — negligible vs the 41 layer all-reduces/forward. One per step, on `FocusRuntimeView.importance` right after the prefix.
- **Determinism:** NCCL all-reduce writes an identical result to every rank, so the post-reduce ΔI is bit-identical ⇒ selection inputs identical. 
  - **(Do the all-reduce in fp32 to avoid bf16 reduction-order drift near top-k ties.)**


## §B — Deterministic selection (no extra sync once §A lands)
- With identical ΔI + identical `avg_decoded` (must stay in sync — it derives from committed counts, which are identical because commits are identical) + identical `mask` (dLLM `input_ids` replicated), `_select_retained` returns identical `retained_maps`/`keep_index`/`new_lens` on every rank **iff the selection itself is deterministic**.
- **Determinism requirements:** top-`target` must break ties by a fixed rule (e.g. lowest block index wins), and the mean/std reductions must be order-stable. 
  - The Plan-A `focus_select_and_enforce_ragged` Triton kernel must be written deterministically (no atomics-with-nondeterministic-order on the selection path). 
  - Verify with a per-rank `(hash(ΔI), |S|, retained_maps)` log asserted identical across ranks each step (see §F).

- **Fallback if determinism is fragile:** select on attention-TP rank 0 and **broadcast** `retain_mask` to the group (one extra small broadcast). Cheap insurance; prefer the deterministic-from-identical-input path and only add the broadcast if a hang is observed.



## §C — Reduced forward under TP (mostly works once §A/§B hold)
- TP attention: each rank computes its head-shard q/k/v + attention on the **same** |S| tokens, then `dense` (RowParallelLinear) all-reduces over |S|; MLP all-reduces over |S|. All ranks share |S| ⇒ lockstep. ✓
- `write_kv`/`set_kv_buffer` is per-rank (head-shard KV) ⇒ no cross-rank. ✓
- Per-phase metadata (`seq_lens`, `kv_indices`, `out_cache_loc`) is identical across ranks (same |S|) ⇒ `init_forward_metadata` consistent; `use_paged=True` is per-rank. ✓
- **Action:** essentially none beyond §A/§B — but add the §F lockstep assertion and run a long TP4 soak to flush any latent divergence.



## §D — Reduced forward under EP (MoE all-to-all)
- L2..L MoE layers dispatch the |S| tokens via EP all-to-all. Identical |S| + identical hidden states ⇒ identical router decisions ⇒ consistent dispatch; fewer tokens (|S|<B) just means a smaller all-to-all, which is already data-dependent on routing. ✓ in eager.
- **Risk (graph/fixed-buffer EP):** SGLang's MoE-EP (and DeepEP) may assume a **fixed/padded token count** for the all-to-all buffers, especially under CUDA graph. Reduced, per-step-varying |S| breaks a fixed buffer.
  - **Mitigation:** reuse the **|S|-bucketization from Plan-A §C** — pad |S| to the same bucket the suffix graph uses, so the all-to-all token count is fixed per bucket. 
  - This couples Plan-A and Plan-B at the bucket ladder (one shared mechanism). Eager EP with dynamic counts can land first; bucketed EP lands with the graph.

- **Action:** verify eager EP4 first (dynamic |S|); then bucket |S| when graphs/DeepEP need fixed counts.





## §E — Reduced forward under DP-attention (hardest; may defer)
- DP-attention shards the **batch** across DP ranks with all-gather/reduce-scatter around attention and a **global** token count for the MLP/MoE (`global_num_tokens_gpu`, `dp_padding_mode`). Selection is per-request (independent), so each DP rank evicts its own requests — fine. **But** the global gather count becomes `Σ_r |S_r|` (sum of reduced counts across DP ranks), which varies per step and per rank-mix.
- **Two sub-problems:** (1) the FOCUS driver bypasses `model_runner.forward`, so it never builds the DP metadata (`global_num_tokens`, padding) — the suffix `ForwardBatch` must carry recomputed `global_num_tokens` = Σ|S_r| (an all-gather of per-rank |S| over the DP group) and the matching `dp_padding_mode`; (2) under CUDA graph the DP path uses a max-len padding mode (model_runner.py:2809) → again needs |S|-bucketing.
- **Action:** **defer DP-attention to a later phase** (TP/EP first, matching the D2 4×A100 TP4/EP4 target). When tackled: thread the reduced `global_num_tokens` into the per-phase `ForwardBatch` and rebuild DP padding, ideally by routing the suffix through a thin `model_runner.forward_focus_*` that reuses the existing DP setup rather than re-implementing it.





## §F — Cross-rank consistency tooling (build before TP runs)
- **Lockstep assertion (env-gated):** each step log per rank `(step, |S| per request, sha1(retain_mask), sha1(ΔI.fp32))`; a small checker asserts all ranks agree. A mismatch is the *exact* precursor to a hang — catch it deterministically instead of via a 300 s watchdog timeout. Use the `debug-distributed-hang` skill methodology (per-rank logging, binary-search the first diverging step).
- **Hang test = correctness test:** run TP4 for many blocks; if any rank diverges in |S|, NCCL hangs — so a clean long soak *is* the proof of §A/§B.



## §G — State persistence across the worker boundary (correctness at serving scale)
- Today `token_sum`/`total_steps` (→ `avg_decoded`, the budget N̄) are tracked **locally within one `Focus.run`** call (the algorithm only sees `ForwardBatch`, not `Req`; see `notes/focus_implementation_progress.md`). Across multi-block serving, N̄ must persist per request. Thread `FocusState` through `ModelWorkerBatch` (the IPC boundary). **Parallelism bonus:** because every rank receives the *same* `ModelWorkerBatch`, the reconstructed per-request state is identical on all ranks ⇒ identical budget ⇒ reinforces §B determinism. Orthogonal to but enabling of correct parallel budgets.



## §H — Validation & ordering
- **Correctness anchor:** α→∞ under TP4 (and TP4/EP4) ≡ LowConfidence under the same config (extend `experiments/dllm/focus_a100_smoke` to TP4/EP4); FOCUS α=1.5 under TP4 coherent.
- **Determinism/hang:** §F lockstep assertion green over a long TP4 soak; per-rank |S| identical every step.
- **Perf:** re-run F1 on the 4×A100 D2 box (TP4/EP4) once correct; compare against LowConfidence TP4/EP4 (and fold in Plan-A graph/kernels).
- **Order:** §A (importance all-reduce) → §F (tooling) → §B/§C (TP, mostly free) → §D (EP eager, then bucketed) → §G (state persistence) → §E (DP-attention, last / optional). §A+§B+§C+§F is the minimal "FOCUS runs correctly on TP" milestone.
- **Coupling to Plan-A:** the |S|-bucketization is shared (graph capture ↔ EP fixed-count all-to-all ↔ DP max-len padding) — design one bucket ladder used by both plans.

Cross-refs: `notes/focus_graph_kernel_plan.md` (shared |S|-bucketization), `notes/focus_paper_exact_plan.md`, `notes/focus_implementation_progress.md`, F1 `experiments/profiling/dllm/focus_vs_lowconf/README.md`; skills `debug-distributed-hang`, `debug-cuda-crash`.