# FOCUS Paper-Exact Reduced Forward — Handoff Plan (for next session)

**Branch:** `feature/focus-implementation`
**Status:** Phase A (logit-masking) landed earlier — saves **zero FLOPs**. Since
then, the host-side math is now paper-correct and the compaction machinery is
built/tested (see §0.1). The remaining work is the split forward + per-phase
attention metadata, specified in §8 (the new, better approach: per-phase
`seq_lens` + contiguous-prefix KV compaction — NO custom Triton kernel, NO
page_size change; R1 resolved). Single A100, LLaDA2.0-mini, TP=1, eager first
(CUDA graph later).

---

## 0.1 Progress update (2026-06-30) — what's now DONE and tested

- **Budget/selection corrected to match the official kernels** (was a real bug):
  - `compute_focus_targets` (focus_utils.py): `target=min(len,max(⌈α·max(avg,1)⌉,1))`
    — **N_σ removed from the budget** (the old `compute_retention_budget` folded
    it in, which was wrong).
  - `compute_should_evict`: `(target>0)&(mask_len>target)`.
  - `select_and_enforce_constraints`: faithful port of
    `focus_select_and_enforce_ragged` — top-`target` by ΔI OR all `ΔI≥mean+std`
    candidates when `≥target` of them (the N_σ expansion), AR-context via block
    adjacency, placeholder integrity gated by `block_progress`, min-keep;
    non-masked processing positions always retained. Returns processing-set
    retain mask + sorted retained block indices.
  - Tests: `test_focus_utils.py`, `test_focus_selection_logic.py` (adds an N_σ
    threshold-OR-topk pinning test + placeholder progress-gate test). Green.
- **State compaction built + tested** (`focus_reduce.py`):
  `build_retained_index` (per-request maps → global gather index + new
  extend_seq_lens), `focus_compact_states` (index_select over flat batch;
  mirrors official fused `focus_compact_states`), `cu_seqlens_from_lens`. Test
  `test_focus_reduce.py` vs double-loop oracle + α→∞ identity. Green.
- `Focus.run` updated to the new targets/should_evict API (still logit-masking
  realization — the split forward in §8 replaces the per-step full forward).

Net: every host-side decision FOCUS makes is now paper-exact and unit-pinned.
The ONLY remaining gap to real FLOPs savings is the split forward (§8).

---

## 0. Why the current Phase A is insufficient (the user's critique, confirmed)

`Focus.run` (current) does: full `model_runner.forward` over all B·batch tokens →
collect importance → select `S` → **mask the logits of non-retained masked
positions** so only `S` may commit. This reproduces the *decoding schedule* but
every transformer layer still runs on all B tokens → **no compute saved**. As a
FOCUS implementation it is hollow. Keep it only as a reference/equivalence oracle
(it equals LowConfidence at α→∞), but the real work is the split forward below.

---

## 1. Official mechanism (ground truth, from ~/FOCUS_ORIGIN)

Files: `lmdeploy/pytorch/models/llada2.py`, `kernels/cuda/focus.py`,
`kernels/cuda/pagedattention.py` (`ragged_paged_attention_fwd`),
`kernels/cuda/fill_kv_cache.py`, `backends/cuda/graph_runner.py`.

**Per denoising step (decode), focus_enabled & evictable:**
1. **Layer 0 — full block** (attn+MLP). During L0 attn, `_compute_focus_importance`
   computes `I0_j` over masked positions via `focus_importance_ragged(q,k,...)`
   and stores it as `context.focus_first_layer_scores`.
2. **Layer 1 — QKV+RoPE on full block**, then:
   - `_prepare_focus_eviction`: compute `I1` (same kernel), `targets = focus_compute_targets(mask_lengths, avg_tokens, alpha)`, `should_evict = (targets>0) & (mask_lengths>targets)`, then `retain_mask = focus_select_and_enforce_ragged(I1, prev_scores=I0, mask_globals, proc_indices, mask_indptr, targets, should_evict, block_progress, max_len)`. ΔI = I1−I0 is formed **inside** the select kernel.
   - **Write FULL-block KV** for L1 via `forward_only_fill_kv(k,v,k_cache,v_cache)` BEFORE eviction (so retained queries can attend to all block keys).
   - `_apply_focus_eviction` → `focus_compact_states(keep_tokens, retain_mask, q,k,v, hidden, input_ids, position_ids, proc_indices, q_lens, new_q_lens, rotary_cos, rotary_sin, residual)`: **gathers all per-token state down to `|S|` tokens** and rewrites `q_seqlens`/`processing_indices`/positions/rotary. Updates `context` via `update_processing_view`.
   - **L1 attention runs on `|S|` queries** (`forward_only_attention`) against the cached KV, then L1 MLP on `|S|`.
3. **Layers 2..L — run entirely on `|S|` tokens** (`LLaDA2PostFocusSuffix.forward`):
   L1-suffix(attn+MLP) then the normal layer loop `for idx in range(2,L)`, final norm.
   The graph_runner captures *this suffix* as a CUDA graph keyed on post-eviction
   token count; the prefix (L0/L1+evict) always runs eager (dynamic shapes).

**KV addressing at L2..L (the crux):** evicted tokens never produce L≥2 KV. The
official suffix attention uses **`ragged_paged_attention_fwd`** with
`tile_to_seq`/`seq_tile_offsets` + `processing_indices` to attend over the
*non-contiguous* retained KV slots (sparse paged attention). KV is **not**
physically compacted into new slots; instead the ragged kernel skips evicted
slots via the processing-index mapping. `fill_kv_cache.py` has the matching
sparse fill. This is the ~900-line Triton path we must port for paper-exactness.

**Budget formula (IMPORTANT — current SGLang code is WRONG here):**
- Official `focus_compute_targets` (focus.py:9-30):
  `target = where(len<=0, 0, min(len, max(ceil(max(avg,1)*alpha), 1)))`.
  **N_σ is NOT in the budget.**
- N_σ lives in the **select** kernel (focus.py:202-213): `mean,std` over ΔI of
  masked positions, `threshold=mean+std`, `candidate_mask=ΔI>=threshold`,
  `use_threshold = (target>0) & (candidate_counts>=target)`; if so select the
  candidate_mask set, else select top-`target` by ΔI. Then AR-context +
  placeholder enforced. So the paper's Eq.4 `max(⌈αN̄⌉,N_σ)` is realized as
  "budget=⌈αN̄⌉; if at least `budget` tokens exceed mean+std, take ALL of them
  (that's the N_σ expansion), else take top-budget."
- **Action:** rewrite `compute_retention_budget` to drop N_σ; move the
  mean+std/threshold logic entirely into `select_and_enforce_constraints`
  (it already half-does this — reconcile so it matches the kernel exactly).

---

## 2. SGLang mapping & integration points (verified this session)

- **Algorithm entry:** `tp_worker._forward_batch_generation_dllm` →
  `self.dllm_algorithm.run(model_runner, forward_batch)` (only `forward_batch` is
  available; no `Req`). Reduced forward must be driven from inside `run` by
  calling new `model_runner`/model methods, not the monolithic `forward`.
- **Model forward:** `LLaDA2MoeModel.forward` (llada2.py:770) is a plain layer
  loop. Add `forward_focus_prefix(...)` (embed→L0 full→L1 QKV+RoPE+fill full KV
  →importance→select→compact to |S|→return hidden,residual,query,retain_meta)
  and `forward_focus_suffix(hidden,residual,query,...)` (L1-attn..L on |S|).
  Mirror on `LLaDA2MoeModelLM`. SDAR later.
- **Importance side-channel:** already added `_collect_focus_importance` on
  `LLaDA2MoeAttention` + `ForwardBatch.focus_view`. REUSE the q,k it already
  has post-RoPE; but for the real path importance must be collected on the
  **uncompacted** L0/L1 q,k (it already is). Keep.
- **Attention backend:** `model_runner.forward_extend` calls
  `self.attn_backend.init_forward_metadata(forward_batch)` (model_runner.py:3213).
  For the suffix we must re-init FlashInfer metadata for the reduced query set
  (q=|S_b| per request, kv=context+block). dLLM uses `is_dllm_extend()` branch
  (flashinfer_backend.py:682,761) with `prefix_lens = seq_lens - block_size`.
  RadixAttention block attention is **non-causal** (ENCODER_ONLY → causal=False,
  flashinfer_backend.py:846-850), good (retained queries see whole block).
- **KV slots:** dLLM `out_cache_loc` covers the full block per request
  (schedule_batch ~1594-1627). For the **sparse** suffix we keep full-block KV
  written at L1 and need a ragged/sparse paged read at L2..L (port of
  `ragged_paged_attention_fwd`). For an **interim compacted** variant (NOT
  paper-exact) we'd write |S| KV to fresh slots and run a normal reduced extend.
- **Logits:** `LogitsProcessor(return_full_logits=True)` already yields
  `full_logits`; for the reduced suffix it returns logits for |S| tokens; scatter
  back to block positions via `retained_to_block_map` before the commit step.

---

## 3. Plan of record (paper-exact, phased)

### Step 1 — Kernels (port from ~/FOCUS_ORIGIN/lmdeploy/pytorch/kernels/cuda/)
Port to `python/sglang/srt/dllm/kernels/focus.py` (new) as Triton, keeping the
PyTorch helpers as numerical oracles for unit tests:
- `focus_importance_ragged` (focus.py:572) — verified axis semantics already.
- `focus_compute_targets` (focus.py:624) — simple; **budget without N_σ**.
- `focus_select_and_enforce_ragged` (focus.py:645 / kernel 142-251) — ΔI=I1−I0,
  mean+std threshold OR top-target, AR-context (retain i−1), placeholder (retain
  masked j<max(S)), min-retain. Unit-test vs current PyTorch `select_and_enforce`.
- `focus_compact_states` (focus.py:676) — gather q/k/v/hidden/residual/ids/pos/
  rotary/proc_indices to |S|, exclusive-prefix-sum keep_offsets. Test vs a torch
  `index_select` reference.
- `ragged_paged_attention_fwd` + sparse `fill_kv_cache` (pagedattention.py:945,
  fill_kv_cache.py) — the hardest. This is what makes L2..L attend the retained
  KV without compaction. **Budget multiple sessions.**

### Step 2 — Split model forward (eager, no graph)
- `LLaDA2MoeModel.forward_focus_prefix`: embed → L0 full → L1 input_ln+QKV+RoPE
  → write full-block KV (need a `save_kv`-only attention call; SGLang RadixAttn
  has no `forward_only_fill_kv` — either add one to the flashinfer backend or do
  a full L1 attention on the full block then discard non-retained outputs for the
  FIRST cut) → importance I0(at L0),I1(at L1) → ΔI/targets/select → compact.
- `LLaDA2MoeModel.forward_focus_suffix`: L1-attn(|S| q vs full-block+context KV
  via ragged paged attn) + L1 MLP → layers 2..L on |S| → norm.
- Thread retained→block index map out so `Focus.run` can scatter logits & commit.

### Step 3 — Wire into `Focus.run`
Replace the per-step `model_runner.forward` with
`model_runner.forward_focus(forward_batch, focus_view)` returning
`(logits_for_S, retained_to_block_map, can_graph)`. Commit decodable among S,
update n_bar/rightmost/uncached. Final full forward unchanged.

### Step 4 — Anchor & perf validation
- **Correctness anchor (unchanged):** α→∞ ⇒ targets=B ⇒ should_evict False ⇒
  |S|=B ⇒ suffix == full forward ⇒ generations identical to LowConfidence
  (reuse `experiments/dllm/focus_a100_smoke`).
- **FLOPs/latency:** log `Σ|S| / (B·batch)` per step (expect ~0.2–0.3, Table 5
  redundancy 15→3) and per-denoising-iter latency drop vs LowConfidence.
- **Quality:** GSM8K/HumanEval coherence with α=1.5.

### Step 5 — CUDA graph (last)
Mirror `graph_runner.py`: eager prefix, capture suffix keyed on rounded |S|
(power-of-2 <256, stride-256 ≥256). Defer until eager perf is proven.

---

## 4. What exists now (this branch)

- `dllm/mixin/req.py`: `FocusState`, `DelayedCacheState` (DC+ neighbor-aware) —
  unit-tested; DC+ is a no-op until reduced forward consumes `uncached_positions`.
- `dllm/algorithm/focus.py`: logit-masking `Focus.run` (NOT compute-saving).
- `dllm/algorithm/focus_utils.py`: `FocusRuntimeView`, `compute_importance_side_channel`
  (axis-verified), `compute_retention_budget` (**has the N_σ-in-budget BUG, fix
  per §1**), `select_and_enforce_constraints`.
- `models/llada2.py`: `_collect_focus_importance` side-channel + `layer_id`.
- `forward_batch_info.py`: `ForwardBatch.focus_view` field.
- Tests: `test_focus_{state,utils,selection_logic,importance_axes}.py` (all pass).
- `experiments/dllm/focus_a100_smoke/`: launch+diff harness (α→∞==LowConfidence).

## 5. First actions next session (ordered)
**Steps 1-2 are DONE (see §0.1). Start at step 3, and follow §8 (the refined
FlashInfer-kv_indices approach) rather than the §3 Triton-sparse-kernel route —
§8 is paper-exact AND avoids porting the 900-line kernel.**
3. (SIMPLIFIED — no new method) KV-fill-only is just a direct
   `forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v,
   layer.k_scale, layer.v_scale)` call (same line the flashinfer backend uses at
   flashinfer_backend.py:887). In the prefix, after L1 QKV+RoPE, call it to write
   the full-block L1 KV; then run L1 attention on the reduced set with
   `save_kv_cache=False`. No backend surgery needed.
4. Build `forward_focus_prefix` (Phase P) / `forward_focus_suffix` (Phases A1+S)
   on LLaDA2 with per-phase FlashInfer `kv_indices` (§8). FIRST resolve the page
   granularity risk (R1: `page_size=block_size` ⇒ block-granular pages; retained
   KV needs token-level indices — confirm the FlashInfer paged wrapper supports it).
5. Wire into `Focus.run`; validate α→∞ == LowConfidence on the smoke harness; log
   `Σ|S|/(B·batch)` redundancy + per-iter latency vs LowConfidence.
6. (Last) CUDA-graph capture of Phase S keyed on rounded |S|.

> The §3/§4 "compacted-KV interim with an L1 approximation" is now SUPERSEDED by
> §8 (custom kv_indices), which is paper-exact with no approximation and no new
> kernel. Prefer §8.

---

## 8. Split forward via per-phase seq_lens + contiguous-prefix compaction (REFINED — 2026-06-30)

**Key discovery:** SGLang's paged attention reads KV through a `kv_indices` page
table built in `init_forward_metadata` (flashinfer_backend.py, the dLLM-extend
branch builds `kv_indptr`/`kv_indices` over `seq_lens`). We can make retained
queries attend EXACTLY the retained KV slots by handing FlashInfer a **custom
kv_indices** that lists only the wanted pages — **no custom Triton sparse kernel
needed** (this supersedes §3's "port ragged_paged_attention_fwd" assumption).
dLLM uses `page_size = block_size`, so pages are block-granular; for true
per-token sparsity the retained-KV phase may need a page_size=1 wrapper or a
per-token kv index. Confirm page granularity first (server_args forces
`page_size=block_size`; may need a FOCUS-specific paged wrapper with token-level
indices, which FlashInfer supports via `paged_kv_indices`/`last_page_len`).

**Why this is paper-exact (no L1 approximation):** the only thing that differs
between the L1-attention phase and the L2..L phase is *which KV slots are listed*
in kv_indices. So run the suffix as TWO metadata regimes:

Per denoising step (decode), focus_enabled & evictable — 3 phases, each its own
attention metadata (reuse `forward_split_prefill`'s `reinit_attn_backend=True`
pattern, model_runner.py:3254, to rebuild metadata between phases):

- **Phase P (prefix): L0 full + L1 QKV.** Metadata = normal full-block dLLM
  extend (q=B, kv=context+B). Run L0 fully (its attention writes L0 KV + collects
  I0 via the existing side-channel). For L1: compute input_layernorm + QKV + RoPE
  on the full block, **write full-block L1 KV to the block's original slots**
  (KV-fill-only — Task 3), collect I1. Do NOT run L1 attention yet. Return
  hidden(B), residual(B), q1(B), I0, I1.
- Host: ΔI, targets, should_evict, select → retained block indices `S`;
  `build_retained_index` → keep_index, new_lens; `focus_compact_states` →
  hidden(|S|), residual(|S|), q1(|S|), positions(|S|).
- **Phase A1 (L1 attention only): q=|S|, kv = context + FULL block (B).**
  Run L1 attention read-only (save_kv=False) against the full-block L1 KV written
  in Phase P, with `paged_kernel_lens = context_len + B` → L1 MLP on |S|.
  Paper-exact: retained queries attend the full block.
- **Phase S (L2..L): q=|S|, kv = context + RETAINED block (|S|).** Each layer
  writes its KV for the |S| retained tokens to the CONTIGUOUS PREFIX of the block
  region (out_cache_loc = first |S| block slots) and reads
  `paged_kernel_lens = context_len + |S|`. Evicted slots are never read
  (paper-exact). Final norm + lm_head on |S|.
- Host: scatter |S| logits back to block positions (retained_to_block_map),
  commit decodable, update n_bar/rightmost/uncached.

**Anchor:** α→∞ ⇒ should_evict False ⇒ |S|=B ⇒ keep_index=identity ⇒ all three
phases collapse to the full forward ⇒ generations == LowConfidence. This is the
gate for the split.

### R1 RESOLVED (2026-06-30) — no custom kernel, no page_size change
SGLang's FlashInfer dLLM extend builds **token-granular** `kv_indices` from
`req_to_token` via `create_flashinfer_kv_indices_triton` — per request it reads a
**contiguous slice** `req_to_token[req, kv_start_idx : kv_start_idx + paged_kernel_lens]`
(flashinfer_backend.py:1416-1431). `paged_kernel_lens` is driven by `seq_lens`
(and `prefix_lens` drives `qo_indptr` = query count). So:
- The retained-KV read is selected purely by **`seq_lens`/`prefix_lens`** per
  phase — NO custom kv_indices, NO page_size=1, NO ragged Triton kernel.
- Requirement: the retained block KV must occupy a **contiguous prefix** of the
  block region in `req_to_token` (so a contiguous slice covers context+retained).
  ⇒ at L2..L, write retained KV to the first |S| block slots (out_cache_loc =
  the block's first |S| token-pool locations). L1's separate cache still holds
  the full B (read with paged_kernel_lens=context+B). Different layers ⇒ no clash.
- Per-phase metadata = standard `init_forward_metadata` with the suffix
  `ForwardBatch` carrying: `extend_seq_lens=|S|`, `seq_lens=context+ (B at A1 | |S| at S)`,
  `positions=compacted`, `out_cache_loc=block-prefix slots`. Reuse
  `reinit_attn_backend=True` (model_runner.py:3254-3261) between phases.
This collapses the earlier "custom kv_indices / sparse kernel" worry into "set
the right seq_lens and write retained KV to a contiguous prefix."

**Implementation order (revised):**
1. (DONE) host math + compaction.
2. KV-fill-only attention call (Task 3): a backend method that writes K,V to the
   cache at `out_cache_loc` without computing attention output. For FlashInfer,
   this is the `save_kv_cache` write path without the wrapper.forward — factor it
   out of `forward_extend`.
3. `LLaDA2MoeModel.forward_focus_prefix` (Phase P) + `forward_focus_suffix`
   (Phases A1+S). Split L1 inside `LLaDA2MoeAttention` into `qkv_rope_fill` and
   `attention_from_q` halves (mirror official forward_focus_qkv_and_evict /
   forward_focus_attention).
4. Per-phase metadata builder: construct the suffix `ForwardBatch`
   (extend_seq_lens=|S|, seq_lens=context+{B|`|S|`}, positions=compacted,
   out_cache_loc=block-prefix slots) and call `init_forward_metadata`
   (reinit_attn_backend). Verify against a 1-request micro-test (q=|S|<B,
   kv=context+B at A1; kv=context+|S| at S) comparing reduced-attention output to
   the full-forward's retained rows.
5. Wire into `Focus.run`; validate α→∞ anchor on the smoke harness; log
   `Σ|S|/(B·batch)` redundancy + per-iter latency vs LowConfidence.
6. (Last) CUDA graph capture of Phase S keyed on rounded |S|.

**Risks:** (R1 — RESOLVED above) token-granular kv_indices via req_to_token ⇒
contiguous-prefix compaction + per-phase `seq_lens`; no kernel/page change needed.
(R2) two metadata rebuilds per step in the hot loop — measure overhead; acceptable
eager, optimize later. (R3) out_cache_loc bookkeeping — write retained KV to the
block's first |S| slots at L2..L; RoPE/positions are per-token (already applied to
q,k before caching), so physical slot order within the block is irrelevant to
correctness under non-causal block attention. (R4) verify FlashInfer extend
accepts q_len(|S|) < kv_block_len(B) at Phase A1 — it should (standard extend
with prefix), but micro-test it (implementation order step 4).
