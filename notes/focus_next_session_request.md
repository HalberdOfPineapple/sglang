# FOCUS — Next Session Request Brief

**Read this first.** This is the single entry point for continuing the FOCUS
implementation. It states the goal, the hard requirements, the authoritative
references, what already exists, the known defects to fix, and the conventions
to follow. Companion deep-dive: `notes/focus_paper_exact_plan.md` (mechanism +
phased plan + exact file/line anchors). Do not start coding before reading both.

---

## 1. The request (what to build)

Implement the **paper-exact FOCUS reduced forward** for SGLang's dLLM path so
that, each denoising step, layers `1-attention .. L` execute on only the
retained token set `|S| ≪ B` (block size `B`), yielding real FLOPs/latency
savings — not just a changed decode schedule.

Target model/hardware for validation: **LLaDA2.0-mini, single A100 80GB, TP=1,
eager first** (CUDA graph is the last phase). SDAR support comes after LLaDA2.

**Definition of done (this milestone):**
1. A split model forward (`forward_focus_prefix` / `forward_focus_suffix`) that
   physically evicts tokens after Layer 1 and runs L1-attn..L on `|S|`.
2. The α→∞ correctness anchor holds: with `alpha` huge, `targets=B`,
   `should_evict=False`, `|S|=B` ⇒ generations are **bit-for-bit identical** to
   `LowConfidence` on the smoke prompts.
3. Measured redundancy `Σ|S| / (B·batch)` per step drops to ≈0.2–0.3 (paper
   Table 5: 15→3) and per-denoising-iteration latency drops vs `LowConfidence`
   at batch ≥ 64.
4. Generation quality with `alpha=1.5` stays coherent (GSM8K/HumanEval spot
   check) — matches paper Table 3/4 trend.

**Non-goals this milestone:** CUDA graph capture (Phase last), multi-GPU (TP/PP),
SDAR. DC+ (delayed cache) only needs to become *active* once the reduced forward
consumes `uncached_positions`; correctness of FOCUS eviction does not depend on
finishing DC+ first.

---

## 2. Hard requirements / invariants (must not break)

- **Correctness before performance** (CLAUDE.md). The α→∞ ⇒ ==LowConfidence
  anchor is the gate; do not optimize past a failing anchor.
- **Isolate the dLLM/FOCUS path.** Do not change AR serving or the existing
  `LowConfidence` path. Gate every new branch on the focus view / a focus flag.
- **Layer roles:** L0 is the only layer that runs full-block attn+MLP. KV for the
  **full block must be written at L1 BEFORE eviction** so retained queries attend
  to all block keys. Evicted tokens never produce L≥2 KV.
- **Block attention is non-causal** (RadixAttention `ENCODER_ONLY` ⇒
  `causal=False`). Retained queries see the whole block.
- **Serial stepping preserved** (`disable_overlap_schedule=True` for dLLM). FOCUS
  per-step state updates depend on it.
- **Single source of importance:** post-RoPE `q,k`. Importance axis semantics
  already verified (MaxPool+Softmax over KEY axis, sum over query+head ⇒ I_j
  indexed by key). Do not re-derive; reuse and unit-test against the kernel.
- **Paper-exact KV read at L2..L:** retained queries attend to context + the
  retained block KV via a **sparse/ragged paged attention** (skip evicted slots
  by index), NOT by physically compacting KV into fresh slots. (A compacted-KV
  interim is allowed only as scaffolding to de-risk plumbing; it carries a
  documented approximation and is exact only at α→∞ — it is NOT the deliverable.)

---

## 3. Authoritative references (ground truth)

Official implementation (LMDeploy) under `~/FOCUS_ORIGIN/`:
- `lmdeploy/pytorch/models/llada2.py` — `forward_focus_prefix` (model:750,
  layer:635), `forward_focus_suffix` (656), `forward_focus_qkv_and_evict` (334),
  `_compute_focus_importance` (220), `_prepare_focus_eviction` (237),
  `_apply_focus_eviction` (285), `LLaDA2PostFocusSuffix` (975).
- `lmdeploy/pytorch/kernels/cuda/focus.py` — `focus_importance_ragged` (572),
  `focus_compute_targets` (624, kernel 9), `focus_select_and_enforce_ragged`
  (645, kernel 142), `focus_compact_states` (676, kernel 339).
- `lmdeploy/pytorch/kernels/cuda/pagedattention.py` — `ragged_paged_attention_fwd`
  (945) with `tile_to_seq`/`seq_tile_offsets`.
- `lmdeploy/pytorch/kernels/cuda/fill_kv_cache.py` — sparse KV fill.
- `lmdeploy/pytorch/backends/cuda/graph_runner.py` — suffix CUDA-graph capture
  (the eager-prefix / captured-suffix split, lines 278-351).
- `~/FOCUS_ORIGIN/notes/code-walkthrough.md` — narrative overview.

Paper: `notes/26_FOCUS.pdf` — Eq. 2 (importance), Eq. 3 (ΔI), Eq. 4/5 (budget,
N_σ), Fig. 5 (design overview), §4.2 (eviction), §4.3 (Intra-Block KV / DC+),
Tables 3–6. Read specific pages with the Read `pages` arg (file is large).

SGLang integration anchors (verified this session):
- Algorithm entry: `python/sglang/srt/managers/tp_worker.py:438`
  (`_forward_batch_generation_dllm` → `dllm_algorithm.run(model_runner, forward_batch)`;
  note: only `forward_batch` is passed, NOT scheduler `Req` objects).
- Model: `python/sglang/srt/models/llada2.py` — `LLaDA2MoeModel.forward:770`,
  `LLaDA2MoeAttention.forward:518`, `_collect_focus_importance` (added).
- Attn backend: `python/sglang/srt/layers/attention/flashinfer_backend.py`
  `is_dllm_extend` branch (682, 761; `prefix_lens = seq_lens - block_size`);
  non-causal at 846-850. Re-init via `model_runner.forward_extend` →
  `attn_backend.init_forward_metadata` (model_runner.py:3213).
- Logits: `LogitsProcessor(return_full_logits=True)` already yields `full_logits`
  (logits_processor.py:247, 1005).
- Positions/KV slots: `forward_batch_info.py:543-554` (dLLM positions),
  `schedule_batch.py:1594-1627` (`out_cache_loc` per full block).

---

## 4. Previous progress (what exists on `feature/focus-implementation`)

Reusable scaffolding (keep):
- `python/sglang/srt/dllm/mixin/req.py` — `FocusState`, `DelayedCacheState`
  (DC+ neighbor-aware). Unit-tested. DC+ is a no-op until reduced forward uses
  `uncached_positions`.
- `python/sglang/srt/dllm/algorithm/focus_utils.py` — `FocusRuntimeView`,
  `compute_importance_side_channel` (axis-verified vs Eq. 2 + kernel),
  `compute_retention_budget` (**HAS BUG, see §5**), `select_and_enforce_constraints`.
- `python/sglang/srt/models/llada2.py` — `_collect_focus_importance` side-channel
  + `layer_id` on attention.
- `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch.focus_view`.
- Tests (run directly with `python <file>`, pytest not installed):
  `test_focus_state.py`, `test_focus_utils.py`, `test_focus_selection_logic.py`,
  `test_focus_importance_axes.py` — all green.
- `experiments/dllm/focus_a100_smoke/` — `run_smoke.sh {focus,focus_alpha_inf,
  low_confidence}` + `drive_smoke.py` + configs + logs. The α→∞==LowConfidence
  diff harness.

To be REPLACED (not deleted until the real path is green):
- `python/sglang/srt/dllm/algorithm/focus.py` — current `Focus.run` is the
  **logit-masking** realization (full forward + suppress non-retained commits).
  Saves ZERO FLOPs. Keep as an oracle; swap its per-step `model_runner.forward`
  for `model_runner.forward_focus(...)` once the split exists.

Commits this branch (newest first): handoff plan; importance-axis verification;
Phase-A validation + smoke harness; importance side-channel + logit-masking;
state structures + helpers; refined plan.

---

## 5. Known defects to fix first

1. **Budget bug.** `compute_retention_budget` folds N_σ into the budget. The
   official `focus_compute_targets` (focus.py:9-30) is
   `target = min(len, max(⌈α·max(avg,1)⌉, 1))` with **no N_σ**. N_σ
   (threshold = mean+std over ΔI of masked positions) is a **selection-time
   expansion** inside `focus_select_enforce` (focus.py:202-213):
   `use_threshold = (target>0) & (candidate_counts >= target)` → if so, retain
   ALL masked tokens with ΔI ≥ mean+std; else retain top-`target` by ΔI. Then
   AR-context (retain i−1) and placeholder (retain masked j < max(S)). Reconcile
   `compute_retention_budget` + `select_and_enforce_constraints` to match the
   kernels exactly, and add a test pinning the "threshold-OR-topk" rule.

---

## 6. Plan of record (ordered; full detail in focus_paper_exact_plan.md §3)

1. Fix the budget/selection split (§5) + tests.
2. Port `focus_compact_states` (gather q/k/v/hidden/residual/ids/pos/rotary/
   proc_indices → |S|) with a torch `index_select` oracle test. Cheapest real
   kernel; unblocks the split.
3. Add a KV-fill-only attention call to SGLang's FlashInfer backend (write KV
   without computing the attention output) — prerequisite for the prefix's
   "write full-block KV at L1 before eviction".
4. Build `forward_focus_prefix` / `forward_focus_suffix` on `LLaDA2MoeModel`
   (+ `LLaDA2MoeModelLM` wrappers). Interim: validate plumbing with a
   compacted-KV reduced extend (documented approximation, exact at α→∞); then
   swap to the sparse `ragged_paged_attention_fwd` + sparse `fill_kv_cache` for
   paper-exactness.
5. Wire into `Focus.run`: per step call `forward_focus`, commit decodable among
   `S`, scatter logits back via `retained_to_block_map`, update
   `n_bar`/`rightmost`/`uncached`. Final full forward unchanged.
6. Validate: α→∞ anchor (smoke harness) + redundancy/latency logging + α=1.5
   quality.
7. (Last) CUDA graph: eager prefix, capture suffix keyed on rounded |S|.

---

## 7. Validation commands

- Unit/logic tests: `python python/sglang/srt/dllm/test_focus_<X>.py`
  (state, utils, selection_logic, importance_axes). No pytest; run files directly.
- End-to-end anchor (1×A100): from `experiments/dllm/focus_a100_smoke/`,
  `./run_smoke.sh focus_alpha_inf` and `./run_smoke.sh low_confidence`, then diff
  `logs/focus_alpha_inf_gen.json` vs `logs/low_confidence_gen.json` (must MATCH).
  `./run_smoke.sh focus` for the α=1.5 quality check.
- Launch flags that MUST be present (single GPU): `--tp-size 1
  --mem-fraction-static 0.7 --disable-cuda-graph --attention-backend flashinfer`;
  model at `/cephfs/shared/model/LLaDA2.0-mini`. Local curl needs
  `--noproxy '*'` / `NO_PROXY` (see memory notes). HF: `HF_HUB_DISABLE_XET=1`.

---

## 8. Noting / output conventions (per CLAUDE.md / AGENTS.md)

- **All notes are Markdown under `notes/`.** Descriptive filenames
  (`focus_<topic>.md`). Compact formatting, no unnecessary blank lines between
  sections; dense but readable; headings + tight bullet lists. Each note must
  carry enough context to be useful later: relevant files/functions, observed
  behavior, open questions, next actions.
- **Experiments** under `experiments/...` with scripts in-repo; profiles/CSVs/
  logs (data) mirror to CephFS `/cephfs/$USER` (or `/cephfs/shared/...`) — never
  leave important artifacts only in `/tmp`, `/root`, or container-local paths.
  Record git branch/commit, model+config, GPUs, parallelism, batch/seq len, and
  the dLLM decoding config (denoising steps, block size, mask ratio, remask/conf
  policy) per the experiment checklist.
- **Commits:** small, targeted, reversible. Branch off `feature/focus-implementation`
  is already active; commit only when a step is green. End commit messages with
  the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- **Distinguish change types** in write-ups (algorithmic dLLM change vs runtime
  scheduling vs distributed comm vs GPU kernel/backend vs memory/cache vs
  measurement) — keep correctness/interpretability first.
- **Update the trail:** when a step lands, update `notes/focus_implementation_progress.md`
  (changelog + checklist) and, if scope/mechanism understanding shifts, the
  `notes/focus_paper_exact_plan.md`. Keep the honesty warning accurate (don't
  call something "validated/complete" if it doesn't save FLOPs).
- **dLLM semantics:** when reasoning, explicitly track masked-token representation,
  whether masked positions attend to each other / decoded tokens, what one
  forward returns, remask/commit selection, and KV-cache reuse validity. Mark
  uncertainty explicitly; verify against code/paper instead of guessing.

---

## 9. Pointers map (one-glance)

| Need | Look at |
|---|---|
| Full mechanism + SGLang hooks + phased steps | `notes/focus_paper_exact_plan.md` |
| Original algorithm recap + per-component plan | `notes/focus_sglang_implementation_plan.md` |
| What's implemented + changelog/checklist | `notes/focus_implementation_progress.md` |
| Official code narrative | `~/FOCUS_ORIGIN/notes/code-walkthrough.md` |
| Paper | `notes/26_FOCUS.pdf` (use Read `pages`) |
| Anchor harness | `experiments/dllm/focus_a100_smoke/` |
| Launch/proxy/HF gotchas | memory: `llada2-launch-config-a100`, `noproxy-localhost-curl`, `hf-download-disable-xet-proxy`, `dllm-eager-needs-flashinfer-attn` |
