# dLLM Distributed Serving — Direction Analysis & Profiling Plan

## 0. Scope and How This Relates to the Project
This note asks: **given the project's two algorithm-level designs, what are the distributed-serving research opportunities on SGLang, and which are genuine innovations vs. pure engineering?** It is downstream of `notes/efficient_dllm_sys.md` (the project designs) and uses the code call-chain in `notes/llada2_workflow_and_parallelism.md` / `notes/dllm_llada2_lowconfidence_launch_walkthrough.md`. FOCUS (`notes/focus_sglang_implementation_plan.md`) is treated only as a **baseline/foil** here, exactly as the deck positions it (§7.3) — not as a direction to chase.

The project's own self-assessment (`efficient_dllm_sys.md` §8) names the gap precisely: the work is *mostly at the algorithm level; system-level design is still limited*, it *introduces no new GPU kernels*, and the central risk (§9.1) is **density ≠ latency** — theoretical sparsity that does not become wall-clock speedup. **That gap is exactly the distributed-serving research surface.** The serving layer is also where two new amplifiers appear that the single-GPU deck does not model: every denoising step fires all TP/EP/DP collectives (comm is `S_k`×-amplified), and per-request sparsity must be *batched* across heterogeneous requests.

## 1. Cost Model (aligned to the deck) + the serving-layer terms
Using the deck's symbols (`efficient_dllm_sys.md` §2.3): `T_dlm ≈ (L/N̄)·C̄`. Design 1 (Streaming Block Controller) raises `N̄` → fewer steps; Design 2 (Draft/Selective Recompute) shrinks `C̄` → cheaper steps. Two serving-layer terms the deck omits:
- **`C̄` is not just FLOPs.** `C̄ = compute(active set) + Σ collectives + host_state_machine + lm_head`. On a distributed runtime the collectives and the host-side state machine can dominate once FLOPs are sparsified — this is the mechanism behind density≠latency.
- **Comm amplification.** Each step's collectives run once per denoising iteration, not per emitted token (`llada2_workflow_and_parallelism.md` §dLLM Notes). So Design 1's "fewer steps" also linearly cuts comm, and Design 2's sparsity *reshapes* comm (smaller, raggeder all-to-all under EP). Comm is therefore a first-class target, not an afterthought.

These notes ground in `LowConfidence` (`python/sglang/srt/dllm/algorithm/low_confidence.py`) as the *current* execution skeleton both designs will replace/extend.

## 2. The Engineering-vs-Innovation Filter
Per the project lead's instruction, classify every direction by one test:
> A direction is an **innovation** only if a **dLLM-specific feature makes it non-trivial** and forces a **new workaround**. If the same solution would work unchanged on an AR serving stack, it is **engineering** — worth doing for deployability, but not a contribution.

Concretely, the dLLM features that create non-triviality are: (a) **bidirectional/full-block attention** (a committed token's validity depends on still-changing future context); (b) **data-dependent, time-AND-depth-varying compute shape** (window size varies per step; active-token density grows per layer, `efficient_dllm_sys.md` §7.2); (c) **the next step's shape is produced by the current step's output** (the change-set drives the next recompute front); (d) **revisable token state** (DRAFT KV can be demoted). Generic AR-serving machinery (uniform 1-token/step, append-only KV, static-shape graphs, head-sharded TP with no per-step consensus) assumes none of these.

## 3. Bucket E — Engineering Optimizations (deployability, not novelty)
These remove overhead but would work identically on an AR stack, so they are not contributions. Still do them — they are prerequisites for measuring the real innovations, and several directly attack density≠latency.
- **E-a Host-sync removal in the step loop.** `LowConfidence.run` calls `.item()` 3–4× per iteration (`low_confidence.py:35,48,53,66,86`) and runs a Python per-block loop (`:59`) — each `.item()` is a device→host sync that serializes the GPU. Fuse selection (argmax/softmax-conf/threshold/commit) into one GPU pass. *Engineering* (a generic kernel-launch/sync cleanup).
- **E-b `lm_head`/full-logits only for live positions.** `full_logits` covers all `block_size*batch` rows over a ~150k vocab every step (`models/llada2.py:803`), but selection reads only masked rows. Gather-then-project. *Engineering.*
- **E-c Static-shape CUDA-graph of the current uniform forward.** Today `input_ids` shape is constant across iterations (in-place mutation), so the *baseline* loop is graphable even though piecewise graph is disabled (`server_args.py:1336`). Check if already on (E8 below). *Engineering — and largely mooted once Design 1/2 make shapes dynamic.*
- **E-d Parallelism-shape tuning for the loop** (TP-only vs TP+EP vs DP-attention). *Engineering tuning.*
- **E-e Plain comm/compute overlap with statically-known shapes.** *Engineering* — but see I3 for the version that is *not*.

## 4. Bucket I — Innovation Directions (non-trivial *because of* dLLM features)
Each item names the dLLM feature that breaks the generic approach, and the new workaround required.

### I1 — Batched execution of ragged, depth-varying active sets
**dLLM feature:** with Selective Recompute the active-Q set is data-dependent *and grows per layer* (density lowest in early layers, rising with depth — `efficient_dllm_sys.md` §7.2); with the Streaming window the per-request length also varies per step. **Why generic batching fails:** AR continuous batching assumes one uniform token per request per step; here every request has a *different active count at every layer*, so a single batched GEMM shape does not exist. **New workaround:** a layout/scheduler that composes per-layer ragged active sets across requests into dense small GEMMs (compaction + indirect indexing), keeping the no-new-kernel constraint by reusing dense kernels on compacted tensors. This is the concrete mechanism that could turn §7's *density* into §5.1's missing *latency*. **This is the highest-value systems contribution candidate.**

### I2 — Distributed selective recompute: agreeing on the active set across MP ranks
**dLLM feature:** the "re-activate KV tokens significantly affected by the change" decision (`efficient_dllm_sys.md` §5.2 step 4) is made *mid-forward* from attention scores. **Why generic TP fails:** TP shards attention heads, so each rank sees only partial scores and may pick a *different* active set → divergent recompute fronts → silent corruption. **New workaround:** a cheap per-layer consensus on the active set (e.g., reduce a `[block]`-sized importance vector over the attention-TP group, or a sharding that makes the decision rank-invariant). Non-trivial because a naive per-layer all-reduce sits in the hot loop and can erase the FLOP savings (quantify with E7-comm) — the research question is *consensus cheaper than the savings it protects*.

### I3 — State-dependent comm/compute overlap (the "overlap" example, the hard version)
**dLLM feature:** the window size and active set for step `t+1` are *produced by* step `t`'s logits+confidence; under EP the all-to-all token counts shrink and rebalance as the active set decays. **Why generic overlap fails:** classic comm/compute pipelining assumes the next op's shape is known ahead of time — here it is not, so you cannot statically schedule the collective. **New workaround:** speculative/adaptive overlap that launches step `t+1`'s communication against a *predicted* active set (Observation 3: next commits are predictable from prior rank — `efficient_dllm_sys.md` §4) and reconciles on misprediction; equivalently, denoising-state-aware dynamic expert load-balancing for the ragged all-to-all. **This is exactly the distinction requested:** plain overlap = E-e (engineering); overlap-under-data-dependent-future-shape = innovation.

### I4 — Revisable-KV cache semantics under paged/radix + distribution
**dLLM feature:** the three-state machine has **DRAFT = soft (revisable) cache** and **UNMASK = frozen cache**, and DRAFT→MASK demotion is allowed (`efficient_dllm_sys.md` §5.2). **Why generic KV cache fails:** SGLang's radix/paged KV is append-only and immutable once committed; a revisable entry that can be invalidated, plus bidirectional attention where a later token's change must invalidate *downstream* cached states, has no equivalent in AR. **New workaround:** cache-coherence rules for revisable entries (when is a DRAFT KV safe to reuse vs. must recompute under bidirectional context), and — distributed — page allocation/eviction/coherence for revisable entries across TP shards. Connects to the deck's open correctness problems (§6) and to dKV-Cache as the baseline.

### I5 — Throughput reframing: pack sparsity across requests, not within one
**dLLM feature:** single-request sparsity does not reduce latency (irregular GEMMs, host overhead — §9.1), but the *freed* compute/comm budget is real. **Why this isn't generic batching:** the budget freed per request is data-dependent and time-varying (it shrinks as a block fills), so admission must track each request's *current* density. **New workaround:** a scheduler that bin-packs heterogeneous, time-varying per-request active densities to keep SM occupancy and the EP all-to-all saturated — i.e., turn the algorithm's FLOP savings into *throughput* even if single-request *latency* is flat. This directly answers §9.1's "density≠latency" by changing the metric that wins.

### Cross-cutting: Design 1 × Design 2 interaction (deck open question §9.3)
Streaming windows change which tokens are in flight, which changes the change-set that drives Selective Recompute. At serving scale this coupling also changes batch composition (I1) and comm shape (I3). Whether the two designs compose or fight is both an algorithm and a systems question — measure jointly, not in isolation.

## 5. Preliminary Profiling Experiments
Run on the working 4×A100 LLaDA2.0-mini launch from `[[llada2-launch-config-a100]]` (`--mem-fraction-static 0.7`, `--max-running-requests 4`). Save traces/CSVs under `/cephfs/$USER/sglang-dllm/outputs/` (AGENTS.md), never `/tmp` or `/root`. Each experiment is tagged **[E]** (informs engineering), **[A]** (informs an innovation direction), or both, with the direction IDs it serves.

### Group A — Characterize the two opportunities at serving scale
- **E1 — Per-step decode-yield `N̄` & front-loading [A: Design 1 / I5].** Instrument the step loop (`low_confidence.py:51–90`) to log, per block, the step count `S_k`, tokens committed per step, and whether the top-1 fallback fired (`:86`). Reproduces the deck's front-loading curve (§3.1) on our checkpoint and tells us how much `N̄` headroom the Streaming Controller can claim. Config-only, do first.
- **E2 — Cross-step / cross-layer redundancy & per-layer density [A: Design 2 / I1].** For a few blocks, dump per-layer the fraction of tokens whose hidden state changed >ε between consecutive steps (cosine-sim of `Q`/hidden, mirroring deck §3.2/§7.2). Output: the **per-layer active-density curve** — the single most important number for I1 (it sets the ragged batch shapes) and for whether Selective Recompute's savings are real on our model.

### Group B — Measure the density→latency gap (the central risk, §9.1)
- **E3 — Sparse-vs-dense forward microbench [E+A: I1].** Outside the server, time one transformer layer (LLaDA2 config) at active-density ∈ {1.0, 0.5, 0.25, 0.1} two ways: (i) full forward + mask, (ii) gather→dense-small-GEMM→scatter. Plot density vs. measured speedup. **This decides whether the no-new-kernel path can realize the sparsity, or whether I1's compaction layout (or kernels, per §8) is mandatory.** The most decision-critical experiment for the project's stated risk.
- **E4 — Host-state-machine overhead [E: E-a].** Torch-profiler/nsys one block; measure GPU idle gaps aligned to `.item()` syncs and the host selection loop. Sizes how much of `C̄` is host overhead vs. compute — i.e., how much density is lost to the runtime before any kernel question.

### Group C — Distributed-specific bottlenecks (the serving amplifiers)
- **E5 — Comm fraction & `S_k`× amplification [A: I3, E: E-d].** Under (a) `--tp-size 4`, (b) `+ --ep-size 4 --moe-a2a-backend deepep`, measure NCCL time / total and per-step collective time (nsys NCCL trace). Confirms comm is `S_k`-amplified and whether EP all-to-all or TP all-reduce dominates — gates whether I3 is worth the hard version.
- **E6 — Ragged EP load imbalance vs. denoising progress [A: I3, I5].** Log per-step the active-token count routed to each expert as a block fills. If the all-to-all becomes imbalanced/under-filled as the active set decays, that is the data-dependent-comm evidence behind I3 and the throughput-packing argument behind I5.
- **E7-comm — Cost of a per-layer active-set consensus [A: I2].** Microbench a `[block_size]`-sized all-reduce over the attention-TP group, ×`num_layers`×`S_k`, at TP∈{2,4,8}. Compare against the per-step FLOP savings from E3. Answers I2's core question: *is consensus cheaper than what it protects?*
- **E8 — Heterogeneous-batch straggler [A: I1, I5].** With `--max-running-requests 4` and a *mixed* easy/hard prompt set, log per request the step its block finishes vs. the step the batch loop exits (`low_confidence.py:52`). Sum of wasted full-batch forwards = the straggler tax that I1's ragged batching and I5's packing must remove.

### Group D — Cheap engineering probes
- **E9 — CUDA-graph A/B [E: E-c].** Check whether the loop replays a captured graph (`can_run_graph` at `low_confidence.py:57,93`); A/B per-step latency with graphs vs `--disable-cuda-graph`. Tells us if E-c is a free win or already on.
- **E10 — Threshold / window sweep [E+A: Design 1].** Config-only sweep of `threshold` (and, once Design 1 lands, window policy); record `S_k`, `N̄`, throughput, and quality on a few GSM8K/HumanEval prompts. The cheapest `S_k`-vs-quality frontier and a baseline for any algorithmic change.

### Suggested order & decision flow
1. **E1, E2, E10** (config-only / lightweight): how much `N̄` headroom and how much per-layer sparsity exists on *our* model. If sparsity is shallow, Design 2's systems work is lower priority.
2. **E3, E4**: the density→latency verdict. If gather→dense-GEMM already realizes the sparsity → I1-via-compaction (no kernels) is viable; if not → kernels are on the critical path and the project's §8 constraint must bend.
3. **E5, E6, E8**: which serving amplifier dominates — comm (`S_k`×), straggler heterogeneity, or EP imbalance — ranking I3/I5 vs I1.
4. **E7-comm**: only if I2 (distributed selective recompute under TP) is on the table.

Decision: **shallow per-layer sparsity** (E2) → focus on Design 1 / I5 (throughput) over Design 2. **Density realizes as latency** (E3) → I1 compaction is the contribution and stays kernel-free. **Comm-dominated** (E5/E6) → I3 (state-dependent overlap) is the contribution. **Straggler-dominated** (E8) → I1/I5 batching. This sequencing avoids committing to a hard systems mechanism before the data says which dLLM feature actually bites at serving scale.

## 6. Instrumentation Notes
- Gate all probes behind `SGLANG_DLLM_PROFILE=1` so the serving path is untouched when off (CLAUDE.md isolation rule).
- NVTX-annotate the step loop and `LLaDA2MoeAttention.forward` / MoE block to separate attention / experts / comm; the built-in `/start_profile`+`SGLANG_TORCH_PROFILER_DIR` may not cover the dLLM loop, so add ranges manually.
- Record git commit, model, GPU count/type, TP/EP/DP shape, block_size, threshold, batch, prompt set per run (AGENTS.md checklist). Write summaries as `notes/experiment_YYYYMMDD_*.md`.

## 7. Open Questions (systems-flavored, feeding the designs)
- Does per-layer sparsity (E2) on LLaDA2.0-mini match the deck's SDAR curves, or is the MoE/active-1.4B model less sparse — changing which design pays off?
- Can gather→dense-GEMM realize the sparsity without new kernels (E3), or is the no-kernel constraint (§8) the thing that must give for a systems contribution?
- Is the per-layer active-set consensus under TP (I2) cheaper than the FLOPs it saves (E7-comm)? If not, Selective Recompute may be TP-hostile and better expressed as DP-replicated.
- Does the freed budget convert to throughput (I5) when single-request latency stays flat — i.e., is the right headline metric tokens/s-per-GPU at fixed quality rather than per-request latency?
- Do Design 1 and Design 2 compose or interfere at batch granularity (§9.3), and does that change the optimal parallelism shape?
