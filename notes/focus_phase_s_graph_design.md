# FOCUS §C — Phase-S CUDA-graph capture (design)

Goal: capture **Phase S** (`forward_focus_rest_and_logits` = L2..L19 + norm + lm_head on the retained
|S| set) as a CUDA graph so the ~62% of wall it costs in eager (per-layer launch latency across 18 MoE
layers on a tiny model — measured by `SGLANG_FOCUS_PHASE_TIMING`, F3-lite) collapses to one replay.
Keep Phase P (prefix) and Phase A1 (L1 attn) eager initially (§C4–C5 capture them later).

## Why Phase S is the hard one to capture
Phase S has a **data-dependent ragged token count** Σ|S_b| (≤ bs·B) that changes every denoising step.
A CUDA graph is a fixed-shape recording, so we must (a) **bucket** Σ|S| to a fixed captured size and
(b) **pad** the real tokens up to the bucket, then (c) rewrite the ragged FlashInfer metadata at replay.
Unlike the existing dLLM graph (which captures the *monolithic* full-block forward keyed on bs, via
`model.forward`, num_tokens_per_bs=block_size — `cuda_graph_runner.py:621`), FOCUS bypasses
`model.forward` and runs the 3-phase split itself, so it needs a **bespoke** capture of just Phase S.

## Bucket ladder (mirror official `graph_runner.py:23,41`)
`phase_s_token_bucket(n)` = powers of 2 up to 256, then stride-256, capped at `max_bs·block_size`:
`1,2,4,8,16,32,64,128,256,512,768,…`. Our regime (B=32, bs≤16 ⇒ Σ|S|≤512) ⇒ buckets {pow2≤256, 512}.
Net win only when `bucket(Σ|S|) ≪ bs·B`; at high redundancy the pad eats the saving (the §C risk — the
plan's bucket-waste vs launch-saving tradeoff; measure + tune the ladder, F3).

## Static buffers (sized to max bucket T_max = max_bs·B), captured once per bucket
Per-token: `hidden[T,h]`, `residual[T,h]`, `input_ids[T]`, `positions[T]`, `out_cache_loc[T]`.
FlashInfer Phase-S metadata (paged prefill wrapper, `use_cuda_graph=True`): `qo_indptr[bs+1]`,
`kv_indptr[bs+1]`, `kv_indices[≤ Σ(context_b)+T]`, `kv_last_page_len[bs]`. The backend already builds
these static buffers for `is_dllm_extend` capture (`flashinfer_backend.py:682`) — but with the
**full-block** regime (q=B, prefix=seq_lens−B). Phase S needs the **reduced** regime
(q=|S_b|, kv=context_b+|S_b|, out_cache_loc=block-prefix), so we add a Phase-S branch / param to the
capture+replay updater (mirror `compute_focus_phase_lens(PHASE_S, …)`).

## Padding scheme (the crux: pad tokens must not corrupt real KV)
Real tokens occupy `[0, Σ|S|)`, request-major; pad tokens `[Σ|S|, bucket)`. Two choices for where the
pad queries live in the ragged layout:
- (rejected after testing) **trailing pad segment** (`bs+1` segments): FlashInfer's cuda-graph wrapper
  **locks the batch size (segment count) at construction**, and an empty pad segment when `pad_len==0`
  (the α→∞ case) is an edge case. Adding a `(bs+1)`-segment wrapper that must also serve `bs` real
  segments mismatches.
- **(CHOSEN) fold pad into the last real request's segment** (`bs` segments, fixed per bs): the last
  request's `qo_len` becomes `|S_{bs-1}|+pad_len`, its `kv` unchanged (context+|S_{bs-1}|). Pad rows
  attend the last request's real KV (non-causal ⇒ valid even if qo>kv; logits sliced off) and write
  their own KV to scratch. `kv_indices` carries ONLY real per-request slices (no pad kv). No empty
  segment ever, and the segment count is exactly `bs` so one **per-bs wrapper** (lazily built, sliced
  to `bs+1` indptr entries) is reused across all buckets — matching how SGLang's own
  `init_forward_metadata_capture_cuda_graph` fixes batch size.
- **FlashInfer cuda-graph constraint (learned by testing):** a `use_cuda_graph=True` wrapper locks BOTH
  its batch size (segment count) AND its **total qo-row count** at the FIRST `plan()` — the qo total
  cannot change afterward (`q.shape[0]` must == `qo_indptr[-1]`). So one wrapper **per (bs, bucket)**
  (qo total == bucket, fixed); priming at a larger qo then planning smaller does NOT work. The kv
  length is likewise bounded by the first plan, but kv **grows across blocks** (context accumulates),
  so we track the captured kv capacity per (bs,bucket) and **re-capture** that graph when a later
  step's live kv exceeds it (context only grows ⇒ re-captures a few times then stabilizes; equal/
  smaller kv just replays). Each distinct (bs,bucket) is captured lazily; pad qo into the last segment
  (Option B). This keeps both qo and the kv-capacity consistent for every replay of a given graph.
**KV-write safety:** every Phase-S layer writes K/V to `out_cache_loc`. Pad tokens' `out_cache_loc` →
a **non-retained block slot the batch already owns** (a block position ≥ |S_b|, never read by Phase S
since the last request's kv range is `[0:context+|S|]`, and overwritten by the block's final full
forward). **No allocator `alloc()`** — that trips SGLang's pool memory-leak detector. Real tokens keep
`build_phase_s_out_cache_loc` (block-prefix). This is the single most important correctness invariant.

## Replay (per denoising step), after host selection
1. `bucket = phase_s_token_bucket(Σ|S|)` (host; Σ|S| from the §A3 `new_lens_cpu`).
2. copy compacted `hidden_s/residual_s/input_ids_s/positions_s` into the static buffers `[0:Σ|S|]`;
   fill `[Σ|S|:bucket]` with a safe pad token/position; set pad `out_cache_loc`→scratch.
3. rewrite Phase-S metadata: `qo_indptr` = cumsum([|S_b| …, pad_len]); `kv_indptr` =
   cumsum([context_b+|S_b| …, pad_len]); `kv_indices` via `create_flashinfer_kv_indices_triton` on the
   padded `req_to_token` slice (+ scratch for the pad segment); `seq_lens=context+|S|`, `prefix=context`.
4. `graph.replay()`; slice `logits[0:Σ|S|]`; `_commit_step` unchanged (consumes the real rows via
   `keep_index`).
This host metadata write is the mandatory **graph break** between the eager prefix and the captured
suffix (§C3); overlap the `new_lens.cpu()` + host arithmetic on a side stream while the GPU finishes
the prefix (the official accepts this exact break).

## Staging (each step validated; α→∞ ≡ LowConfidence + graph-on == graph-off)
1. **C-foundation (this commit):** `focus_graph.py` — `phase_s_token_bucket` + ragged
   pad-layout builder (`build_phase_s_graph_layout`: padded qo/kv segment lens + out_cache_loc with
   pad→scratch + real_token_count). Pure tensor, unit-tested vs oracle. No graph yet.
2. **C-microtest ✅ DONE (2026-06-30, `test_focus_phase_s_graph_gpu.py`, A100):** captured a CUDA graph
   around the FlashInfer paged-prefill Phase-S call (`use_cuda_graph=True`, plan-outside / run-inside),
   replayed with a smaller real |S| padded to the bucket (pad queries → scratch slot 0), real rows match
   the eager non-causal oracle (max err 4.3e-3); graph **reused** with new q content (re-plan + replay)
   still correct (4.6e-3). **De-risked: FlashInfer-in-graph works, and the pad/scratch-KV invariant
   holds.** The keystone §C risk is retired at the attention level.
3. **C-capture:** a `FocusPhaseSGraph` runner capturing `forward_focus_rest_and_logits` per (bs,bucket),
   with the static buffers + the Phase-S metadata capture/replay branch in `flashinfer_backend.py`.
4. **C-wire:** `Focus._focus_reduced_forward` replays the graph for Phase S when a graph for
   (bs, bucket) exists, else eager fallback. Re-run F1; F3 to confirm s_fwd device-time drops.
5. **C4–C5:** capture Phase P (existing dLLM bs-buckets) and fold A1 into the Phase-S graph (second kv
   regime) or keep eager.

## Risks
- Bucket waste at high redundancy (pad → bs·B); tune ladder, measure (F1/F3).
- FlashInfer paged-prefill inside a CUDA graph with rewritten ragged indptr — de-risked by C-microtest.
- Scratch-KV slot must be reserved out of the allocator (or use an always-safe existing slot); a wrong
  scratch loc silently corrupts a real block. Highest-severity invariant.
- dLLM currently runs eager (`dllm-eager-needs-flashinfer-attn`); enabling a partial graph must not
  perturb the eager prefix/A1 or normal LowConfidence serving.

Cross-refs: `focus_graph_kernel_plan.md` §C, `focus_implementation_progress.md` (session 3b timing),
`flashinfer_backend.py:561,682,714,761`, `cuda_graph_runner.py:595,621`, official
`~/FOCUS_ORIGIN/lmdeploy/pytorch/backends/cuda/graph_runner.py`.
