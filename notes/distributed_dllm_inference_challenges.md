# Distributed dLLM Inference — Unique Challenges vs Conventional LLM Serving

## 0. Scope and framing
This note asks one question in two halves: **(A)** what makes *distributed, multi-GPU* dLLM inference structurally harder than distributed AR inference, independent of any project change; and **(B)** under the project's two algorithmic contributions (`notes/efficient_dllm_sys.md`: *Streaming Block Controller* and *Draft-Token State + Selective Recompute*), what *new* distributed challenges appear as we go from 1 GPU → multi-GPU → multi-node. It sits between `notes/llada2_workflow_and_parallelism.md` (the TP/EP/PP/DP call chain), `notes/dllm_distributed_optimization_directions.md` (directions I1–I5, the engineering-vs-innovation filter), and the measured baseline in `experiments/profiling/dllm/d2_sk_amplification/` (D2). Where a claim is measured, it is anchored to D2; where it is a design consequence not yet measured, it is flagged *(design)*.

The single thesis: **AR distributed serving is built around one fact — one forward emits one token per request, append-only, with a static shape — and almost every SGLang distributed mechanism (overlap scheduler, continuous batching, paged/radix KV, CUDA graphs, head-sharded TP) silently assumes it.** dLLM violates that fact in four independent ways, and each violation breaks a different distributed assumption. The project's algorithms then *deepen* the violations (dynamic shapes, ragged active sets, revisable cache), trading single-GPU FLOPs for distributed-coordination cost — which is exactly where the "density ≠ latency" risk (`efficient_dllm_sys.md` §9.1) lives.

---

## Part A — Why distributed dLLM differs from distributed AR (baseline, no project change)

### A1. Collectives are `S_k`×-amplified — comm scales with *steps*, not *tokens*
In AR, one forward → one token, so one round of TP all-reduce / EP all-to-all per emitted token. In dLLM, a block of `B=32` tokens is denoised over `S_k` iterative forwards, and **every** forward fires the *full* set of TP/EP/DP collectives over the *whole* block (`llada2_workflow_and_parallelism.md` §dLLM Notes; `low_confidence.py:71`). D2 measures `S_k ≈ 15` forwards per 32-token block on LLaDA2.0-mini/HumanEval. The penalty has to be stated carefully — it is a **volume** penalty, not a round-count penalty:
- **Round count is *not* amplified per token.** 15 forwards deliver 32 tokens = **0.47 collective rounds per delivered token**, *fewer* than AR's 1/token. The "**~15× round amplification**" D2 reports is **per block vs the diffusion-native ideal** of 1 round/block (all 32 resolved in a single parallel step, `N̄ = B ⇒ S_k = 1`) — it is the *Design-1 "fewer steps"* target, not a per-token or a vs-AR number.
- **Volume *is* `S_k`×-amplified per delivered token vs AR.** Because each of the 15 forwards re-processes the entire block, the all-reduce moves `[bs×B, hidden]` *every* step (`S_k × B` token-passes per block = `S_k` token-passes per *delivered* token), vs AR decode's one token's worth of hidden state per collective. So dLLM pushes **~`S_k` ≈ 15× the collective *volume* per delivered token** — D2's `comm_per_output_token = (S_k × comm_per_step) / tokens_committed`.

**The crux — comm volume *per forward*, the easy thing to get wrong.** The whole comparison hinges on *how many tokens flow through the all-reduce in one forward*, and AR vs dLLM differ by `B`:
- **AR decode forward = 1 token of comm.** The KV cache means only the single new token is projected through the linear layers, so the TP all-reduce is over `[bs×1, hidden]` — measured shape `[bs×1, 2048]`.
- **dLLM forward = `B`=32 tokens of comm.** The active block has **no KV cache** (its contents are still changing — `efficient_dllm_sys.md` §2.2; attention is `ENCODER_ONLY`, A4), so **all 32 block positions are re-projected every step** and the all-reduce is over `[bs×32, hidden]` — the **measured** D2 shape `[bs×32, 2048]`. A dLLM forward therefore carries **32× the comm of an AR decode forward**, not 1×.

Counting comm to deliver one 32-token block makes the `S_k` factor fall out:

| | # forwards | tokens through all-reduce **/forward** | total comm (token-units) | per delivered token |
|---|---|---|---|---|
| **AR decode** | 32 | **1** (KV cache → only new token reprojected) | 32 × 1 = **32** | 1 |
| **dLLM** | `S_k`=15 | **32** (no KV for active block → whole block reprojected) | 15 × 32 = **480** | **15** |

Ratio `480/32 = 15 = S_k`. The common mistake is to model dLLM as "15 forwards of 1 token = 15 units" (which would be *less* than AR's 32) — but a dLLM forward is "1 forward of the whole 32-token block," so it is `15 × 32 = 480`. dLLM doesn't do fewer-forwards-of-one-token; it does `S_k` forwards of the *entire block*. This is the same `S_k × B` token-forwards that drive the deck's "dLLM costs several× more **compute** than AR" (`efficient_dllm_sys.md` §2.3) — comm and compute are amplified by the *same* factor because both scale with token-forwards. **If the active block kept a usable KV cache** (each forward reprojecting only the newly-changed positions instead of all 32), the per-forward volume would collapse toward 1× and the 15× would vanish — which is exactly what Design 2's selective recompute tries to buy. The *baseline* reprojects the full block every step, which is what D2 measured.

The amplification is **content-driven and interconnect-independent** (intrinsic `s_k` mean 14.7/13.7/14.9/15.0 across bs 1/4/8/16, matching the H100 sibling). This volume penalty is the headline distributed cost of diffusion decoding and the comm-side motivation for Design 1 (fewer steps lowers `S_k`, cutting both the per-block rounds *and* the per-token volume linearly).

### A2. The comm is *fully exposed* — overlap scheduling is force-disabled
AR serving hides most collective time behind compute via the overlap (future-token) scheduler. For dLLM that scheduler is **force-disabled** (`server_args.py` `_handle_dllm_inference`, `disable_overlap_schedule=True`) because the denoising loop mutates `forward_batch.input_ids` *in place* mid-step (`low_confidence.py`), so the next step's input is not known until the current step's logits land. Consequence: **all `S_k`× comm is exposed wall-clock**, the opposite of AR. D2 (H100, real NVLink) measures the exposed collective fraction at **22–32%** of the forward; the A100 bandwidth-floor projection is ≥7.7% and a latency-aware estimate ~14%. The lever this points to is collective *count/latency* and *exposure*, not interconnect bandwidth — bandwidth is explicitly *not* the first-order dLLM lever (D2 A100 takeaways).

### A3. The EP all-to-all moves *full-block* volume every step
Baseline does no token eviction, so the MoE dispatch/combine all-to-all under EP moves `B·batch` tokens **every** denoising step, not the `1·batch` of AR decode. The all-to-all *volume* is therefore `S_k`× larger for the same output. Worse, the block *content* changes step-to-step as masks resolve, so **expert routing drifts across the `S_k` steps even though the token count is constant** — a dLLM-specific load-imbalance pattern with no AR analog (D4 in the plan; the one genuinely rank-divergent behavior in the stock path). On the A100/no-NVLink box this comm-boundedness produces **flat throughput across batch size (74/75/70/69 tok/s)** — batching buys *zero* scaling because comm grows ∝ bs while PCIe bandwidth is fixed.

### A4. Bidirectional / encoder-only attention breaks AR's KV invariants
LLaDA2 runs `RadixAttention` in `AttentionType.ENCODER_ONLY` (`llada2.py:508`): all positions in a block attend to each other and to the cached prefix, and **masked positions are overwritten in place across iterations**. Two distributed consequences: (i) a committed token's validity depends on *still-changing future context within the block*, so the usual AR guarantee "once written, KV is immutable" does not hold for the in-flight block — KV-cache reuse needs explicit reasoning (`llada2_workflow_and_parallelism.md` §dLLM Notes); (ii) `page_size` is force-set to `block_size` (32) so KV is allocated/freed at block granularity, a coarser paging than AR's per-token growth, which changes per-rank KV residency and fragmentation under distributed batching (D8). The radix prefix only advances one *finished* block per round (`scheduler.py:2512` stash), so cross-step KV coherence is confined to the active block — but that confinement is what the project's Design 2 will deliberately break (Part B).

### A5. Lockstep batching → distributed straggler tax
A decode batch of N blocks denoises in **lockstep**: each iteration runs one `model_runner.forward()` over the whole `N·B` tensor, and the loop runs until the *slowest* block is mask-free (`low_confidence.py:71`). A finished block is skipped only in the post-forward *selection* (`continue`, `low_confidence.py:92`) — it is still dragged through every forward *and every collective*. So batch heterogeneity wastes not just FLOPs but **whole collective rounds**: D2 measures straggler waste rising **0 → 36 → 39 → 46%** of each block's forwards as bs grows 1→16. AR continuous batching avoids this by letting each request advance independently; dLLM cannot, because the forward is one fused shape. This is the distributed-cost framing of the straggler problem and the motivation for ragged batching (I1) and throughput packing (I5).

### A6. Phase-pure, prefill-first scheduling — no chunked-prefill mixing
AR serving deliberately *mixes* prefill and decode in one batch (chunked prefill) to keep the GPU saturated. dLLM **cannot**: rounds are phase-pure and prefill wins (`mixin/scheduler.py:149`), because a no-mask prefill block mixed into a decode batch would lose the one-forward fast path and be re-forwarded `(S_k+1)×` for nothing (`llada2_workflow_and_parallelism.md` §"Why phase-pure"). Consequence at serving scale: a long prompt forces `ceil(prompt_len/B)` prefill rounds that each **stall all decode requests and waste their collective rounds** — a dLLM-specific prefill/decode interference that AR's mixing was designed to remove.

### A7. Rank divergence is confined — but consensus assumptions still differ
Under TP/EP all ranks run the *same* denoising schedule on the *same* batch; the only genuinely rank-divergent behavior in the stock path is **MoE expert routing under EP** (D4). This is actually *easier* than AR in one respect (the accept/commit decision reads full logits that are gathered, so the decision is rank-consistent by construction). But it sets a trap for Part B: the moment Design 2 makes the *active set* a per-rank, mid-forward decision, this benign uniformity is gone (I2).

### A8. Multi-node makes A1–A3 brutal, and overlap can't save you
Inter-node NCCL is far slower than intra-node NVLink, and the `S_k`× amplification (A1) multiplies that slower hop ~15×, with **none of it hideable** because overlap is disabled (A2). PP is *forced to 1* for dLLM (`server_args.py` override), removing the one parallelism axis that tolerates slow links via pipelining — so a multi-node dLLM deployment must lean on TP/EP across nodes, exactly the collective-heavy axes. D9 in the plan flags this as a possible hard constraint: **stock dLLM serving may not be viable multi-node at all without step reduction or overlap**, whereas AR routinely spans nodes.

| # | Distributed assumption AR relies on | How dLLM breaks it | Measured anchor (D2) |
|---|---|---|---|
| A1 | comm ∝ tokens emitted | comm ∝ `S_k` steps/block, full-block each step | ~15× *volume*/token vs AR (rounds/token 0.47, fewer); 15× rounds/block vs 1-round ideal |
| A2 | overlap hides comm | overlap force-disabled, comm exposed | 22–32% (H100), ~14% (A100 lat.) |
| A3 | all-to-all moves 1 token/req/step | full-block volume every step; routing drifts | flat 70 tok/s on PCIe |
| A4 | KV immutable once written | in-place mask overwrite, bidirectional, page=block | — |
| A5 | requests advance independently | lockstep until slowest block | straggler waste →46% |
| A6 | mix prefill+decode (chunked) | phase-pure, prefill-first | — |
| A7 | ranks uniform except MoE routing | (baseline holds) | — |
| A8 | PP + overlap tolerate slow links | PP=1, overlap off, `S_k`× on slow hop | D9 (todo) |

---

## Part B — New distributed challenges introduced by the project's algorithms

The two designs are framed as *single-GPU* algorithmic wins (no new kernels, decision made before each step — `efficient_dllm_sys.md` §8). The distributed surface is precisely where those wins meet friction, and several Part-A challenges *invert* (e.g. dynamic shapes make A4's static graph illegal; ragged active sets make A5's straggler a per-layer problem). Mapped to the directions I1–I5 in `dllm_distributed_optimization_directions.md`.

### B1. Design 1 (Streaming Block Controller) — dynamic shapes vs the distributed runtime
The controller resizes the decoding window per step (the deck cites 32 → 80+ positions; `efficient_dllm_sys.md` §5.1). Distributed frictions:
- **CUDA-graph illegality.** Baseline dLLM is graphable *because* `input_ids` shape is constant across iterations (in-place mutation, A4). A per-step-variable window makes the forward shape dynamic → graph replay buckets no longer match → either re-capture cost or fall back to eager, which *re-exposes* per-launch overhead and (per the eager gotcha in the experiment README) inflates NCCL via cross-rank spin-wait. **Fewer steps (good for A1 comm) can cost the graph (bad for launch/exposed comm)** — these must be measured jointly, not assumed additive.
- **Comm shape varies per step.** A larger window = more tokens through every TP all-reduce and EP all-to-all that step. Design 1's "fewer steps" cuts the *number* of collective rounds (directly attacks A1) but *raises the per-round volume*; the net comm depends on whether `Σ window_t` over the streamed schedule is below `S_k_baseline × B`. *(design — needs the D2 counter re-run under a streaming policy.)*
- **Batch composition coupling.** Per-request window size now varies, so the lockstep batch (A5) has rows of *different active length* even before Design 2 — the fused `N·B` shape stops existing. This is the entry point for I1 (ragged batching).
- **Multi-node positional dependency.** §2.4 notes output length must be fixed before sub-tasks parallelize (positional encoding imposes strict position dependency). A streaming window that grows across a block boundary changes `dllm_block_offset`/RoPE placement (`forward_batch_info.py:538`); coordinating that window-growth decision *consistently across ranks and nodes* is a new consensus the baseline never needed.

### B2. Design 2 (Selective Recompute) — ragged, depth-varying active sets across ranks [I1]
Selective Recompute drives a sparse active-Q set that **grows per layer** (density lowest in early layers, rising with depth — `efficient_dllm_sys.md` §7.2). Distributed frictions:
- **No single batched GEMM shape exists.** AR continuous batching assumes one uniform token/request/step; here every request has a *different active count at every layer*. Composing per-layer ragged active sets across requests into dense small GEMMs (compaction + indirect indexing) is the highest-value systems contribution candidate (I1) — and the concrete mechanism that could turn D2/§7 *density* into the missing *latency*.
- **Density ≠ latency is a distributed problem, not just a kernel one.** Even if a compacted GEMM realizes the sparsity locally, the collectives (A1) and the host-side state machine still run per step. `C̄ = compute(active) + Σ collectives + host_state_machine + lm_head` (`dllm_distributed_optimization_directions.md` §1): sparsifying only the first term while the others stay fixed is exactly how theoretical sparsity fails to become wall-clock speedup at serving scale.

### B3. Design 2 — distributed consensus on the active set under TP [I2]
The "re-activate KV tokens significantly affected by the change" decision (`efficient_dllm_sys.md` §5.2 step 4) is made **mid-forward, from attention scores**. Under TP, attention heads are sharded (`llada2.py:440`), so **each rank sees only partial scores and may select a *different* active set → divergent recompute fronts → silent cross-rank corruption**. This is the dangerous inversion of A7: the baseline's benign rank-uniformity is gone. The required workaround is a cheap per-layer consensus (e.g. reduce a `[block_size]`-sized importance vector over the attention-TP group, or a sharding that makes the decision rank-invariant). The research tension: a naive per-layer all-reduce sits in the hot loop ×`num_layers`×`S_k` and can **erase the very FLOP savings it protects** (I2; quantify against D2's per-step savings, plan E7-comm). If consensus is too expensive, Selective Recompute may be **TP-hostile and better expressed as DP-replicated attention** — a parallelism-shape decision forced by the algorithm.

### B4. Design 2 — revisable KV cache coherence under paged/radix + distribution [I4]
The three-state machine has **DRAFT = soft (revisable) cache** and allows **DRAFT→MASK demotion** (`efficient_dllm_sys.md` §5.2). SGLang's radix/paged KV is **append-only and immutable once committed** (A4) — there is no AR equivalent of an entry that can be invalidated. Distributed frictions:
- **Coherence rules for revisable entries.** When is a DRAFT KV safe to reuse vs. must-recompute under bidirectional context? A later token's change must invalidate *downstream* cached states — a write-invalidation pattern foreign to AR's monotone cache.
- **Distributed page allocation/eviction.** Across TP shards (and across nodes), a revisable page that one rank demotes must be coherently invalidated everywhere it is mirrored or referenced. This connects to the deck's open correctness problems (§6: "one token can trigger global attention updates") and to dKV-Cache as the baseline.

### B5. Design 2 — confidence is unreliable under sharded / partially-frozen context
`efficient_dllm_sys.md` §6 already flags that confidence from draft/unmasked inputs cannot be trusted at face value. Distribution compounds this: the accept/promote decision (`DRAFT→UNMASK`, *irreversible*) is taken against context that is both *partially frozen* (algorithmic) *and partially sharded* (TP). The deck's mitigations — "accept only when multiple steps agree" (temporal consistency, Observation 3) and offline-calibrated reliable confidence — must hold **rank-consistently**, or different ranks freeze different tokens (the B3 corruption mode applied to the irreversible transition, where it is unrecoverable).

### B6. Throughput reframing — pack sparsity across requests, not within one [I5]
If single-request sparsity does not cut latency (B2), the *freed* compute/comm budget is still real but **data-dependent and time-varying** (a block's freed budget shrinks as it fills). A scheduler that bin-packs heterogeneous, time-varying per-request active densities to keep SM occupancy and the EP all-to-all *saturated* turns the algorithm's FLOP savings into **throughput** even when per-request latency is flat — directly answering the density≠latency risk by changing the winning metric to tokens/s-per-GPU at fixed quality. This also *fixes* A5/A6: ragged packing removes the lockstep straggler tax and the phase-pure stall. *(design.)*

### B7. Multi-node: every Part-B friction gets a slow hop and no overlap
Going from multi-GPU to multi-node turns each Part-B coordination into an inter-node collective on top of A8's already-brutal baseline:
- **Per-layer active-set consensus (B3)** becomes an *inter-node* all-reduce ×`num_layers`×`S_k` — almost certainly more expensive than the FLOPs it protects, pushing Selective Recompute toward intra-node-only TP groups with DP-attention across nodes.
- **Revisable-KV invalidation (B4)** must cross node boundaries coherently — a distributed-cache-coherence problem at NCCL latencies, paid per demotion.
- **Streaming-window / active-set shapes (B1)** for step `t+1` are produced by step `t`, so the inter-node collective for `t+1` cannot be statically scheduled — the hard version of comm/compute overlap (I3): speculative/adaptive overlap that launches against a *predicted* active set (predictable per Observation 3) and reconciles on misprediction. Plain overlap = engineering; overlap-under-data-dependent-future-shape across nodes = the genuine innovation.
- **Net:** the project's algorithms are *most* defensible as **intra-node** mechanisms (TP/EP within a node, DP/request-replication across nodes), precisely because the consensus and cache-coherence they introduce are latency-sensitive. Whether they compose across nodes at all is the open distributed-systems question.

| # | Design | New distributed challenge | AR analog? | Direction |
|---|---|---|---|---|
| B1 | Streaming window | dynamic shape breaks CUDA graph; per-step comm volume varies; cross-rank window consensus | none (AR static shape) | I1, I3 |
| B2 | Selective Recompute | ragged depth-varying active sets → no single GEMM; density≠latency incl. comm+host | none (AR uniform 1 tok) | I1 |
| B3 | Selective Recompute | mid-forward active-set choice diverges across TP shards → corruption | none (AR rank-uniform) | I2 |
| B4 | Draft state | revisable/invalidatable KV vs append-only radix; cross-shard coherence | none (AR immutable KV) | I4 |
| B5 | Draft→Unmask | confidence unreliable under sharded+frozen context; irreversible freeze must agree across ranks | none | I2 |
| B6 | both | pack time-varying sparsity across requests for throughput | partial (continuous batching) | I5 |
| B7 | both | every B-challenge on a slow inter-node hop, no overlap, PP=1 | none | I3, I2, I4 |

---

## 3. Bottom line
- **Baseline distributed dLLM is already harder than AR** for four independent, measured reasons: `S_k`× exposed comm (A1/A2), full-block all-to-all with routing drift (A3), in-place bidirectional KV (A4), and lockstep straggler waste (A5). The first-order lever is **collective count × `S_k` and its exposure**, not interconnect bandwidth (D2).
- **The project's algorithms trade single-GPU FLOPs for distributed coordination cost.** Streaming windows break the static-shape assumptions that made the baseline graphable (B1); Selective Recompute introduces ragged cross-rank active sets (B2), a mid-forward consensus problem under TP (B3), and revisable-cache coherence (B4) — none of which exist in AR.
- **"Density ≠ latency" is fundamentally a distributed claim:** the freed FLOPs sit next to fixed-cost collectives and a host state machine, so sparsity only becomes speedup if the *layout* (I1) and *comm* (I3) are co-designed — and only safely if cross-rank consensus (I2) and cache coherence (I4) are cheap enough.
- **Multi-node is the stress test.** With PP forced off, overlap disabled, and `S_k`× amplification on a slow hop, the algorithms are most defensible **intra-node**, with request-DP across nodes. Whether Selective Recompute's per-layer consensus and revisable-KV coherence survive inter-node latency is the central open distributed-systems question (B7, plan D9).

## 4. Open questions (feeding the profiling plan)
- Does a streaming window's *fewer steps* (less A1 comm) outweigh its *larger per-step volume* and *lost CUDA graph* (B1)? — re-run D2's `S_k`/comm counter under a streaming policy.
- Is per-layer active-set consensus under TP (B3) cheaper than the FLOPs it saves (plan E7-comm)? If not, Selective Recompute is TP-hostile → DP-replicated attention.
- Can gather→dense-GEMM realize Selective Recompute's sparsity without new kernels (plan E3), or does the no-kernel constraint (§8) have to give for a *distributed* speedup?
- Does the freed budget convert to throughput (B6/I5) when single-request latency stays flat — is tokens/s-per-GPU at fixed quality the right headline metric?
- Do Design 1 and Design 2 compose or fight at *batch* granularity (changing both A5 stragglers and A3 comm shape), and does that change the optimal parallelism shape (§9.3)?
