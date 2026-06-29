# FOCUS Paper-Exact Reduced Forward — Handoff Plan (for next session)

**Branch:** `feature/focus-implementation`
**Status:** Phase A (logit-masking) landed + validated, but it saves **zero FLOPs**
(runs all B tokens through every layer; only gates which positions may commit).
This note specifies the **paper-exact reduced forward** that actually evicts
tokens after Layer 1 so layers 1-attn..L run on `|S| ≪ B` tokens. Single A100,
LLaDA2.0-mini, TP=1, eager first (CUDA graph later).

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
1. Fix budget/selection split to match official kernels (§1 budget bug). Re-run
   `test_focus_selection_logic` + add a test pinning the kernel's
   "threshold-OR-topk" rule.
2. Port `focus_compact_states` + a torch oracle test (cheapest real kernel).
3. Add `forward_only_fill_kv`-equivalent to SGLang flashinfer backend (write KV
   without computing attention output) — prerequisite for prefix.
4. Build `forward_focus_prefix/suffix` with the **compacted-KV interim** first
   (writes |S| KV to fresh slots, standard reduced extend) to get the split +
   anchor green and measure real latency drop; THEN swap to sparse ragged paged
   attention for paper-exactness (no L1 full-block-KV approximation).
   - NB: the compacted-KV interim has ONE approximation (retained attend retained
     at L1, not full block). It is exact at α→∞. Use it only to de-risk the split
     plumbing; the paper-exact target is the ragged sparse path in step 1/§3.
