# Distributed dLLM Training — Unique Challenges vs Conventional LLM Training

## 0. Scope and framing
This note extends `distributed_dllm_inference_challenges.md` to the training regime. Where inference asks "how does distributed *serving* differ for dLLMs vs AR," training asks "how does distributed *pre-training/fine-tuning* differ for dLLMs vs AR." The core tension remains: **AR training is built around one fact — one forward emits one loss term per token, with gradient flowing back through a causal, append-only dependency chain — and dLLM violates that fact through iterative denoising, bidirectional dependencies, and mask scheduling.** Each violation creates distributed friction in backward passes, gradient synchronization, optimizer coordination, and checkpointing that AR training never encounters.

The training question has both a **backward-pass mirror** of inference challenges (Parts A/B become T-A/T-B) and **genuinely new training-only problems** (gradient accumulation semantics, optimizer state distribution, checkpoint bloat, curriculum effects on convergence) that have no inference analog. Unlike inference where the project's algorithms (Streaming Block Controller, Selective Recompute) introduce specific new distributed costs, training challenges arise from the **base dLLM formulation itself** — mask scheduling, confidence-based remasking, and non-autoregressive loss.

---

## Part T-A — Why distributed dLLM training differs from distributed AR training (baseline)

### T-A1. Backward pass is `S_k`×-amplified — gradient comm scales with denoising steps, not output tokens
**Forward mirror:** Just as inference collectives fire `S_k` times per block (A1), training backward passes and gradient all-reduces fire `S_k` times per block. For a 32-token block with `S_k=15` denoising steps:
- **AR training:** 32 forward passes → 32 backward passes → 32 gradient all-reduces (one per emitted token, causal chain).
- **dLLM training:** `S_k=15` forward passes over the full 32-token block → 15 backward passes → 15 gradient all-reduces over **full-block activations**.

**Volume amplification:** Each dLLM backward propagates gradients through `[bs×B, hidden]` activations (the entire block, A1), not `[bs×1, hidden]` (one new token). So gradient all-reduce volume per delivered training token is:
```
gradient_comm_per_token = (S_k × grad_size_per_step) / B
                        = (15 × [bs×32×hidden]) / 32
                        = 15 × [bs×hidden]
```
vs AR's `1 × [bs×hidden]` per token. **Factor: ~15× gradient communication volume per output token.**

**Critical difference from inference:** Inference overlap-disable (A2) was algorithmic (next input unknown until current logits land). Training overlap-disable is **intrinsic to backprop semantics** — you cannot start step `t+1`'s backward until step `t`'s backward completes and gradients accumulate, because they share parameter gradients. So the `S_k`× gradient comm is **always exposed in the backward critical path**, even if you could overlap forward passes.

### T-A2. Gradient accumulation semantics — how do `S_k` denoising steps map to one optimizer step?
AR training: one microbatch forward → one backward → accumulate gradients → optimizer step after `N` microbatches.

dLLM training has **nested accumulation**:
1. **Inner loop (per block):** `S_k` forward/backward pairs over one block, all sharing the **same input block but with changing masks**. Do gradients accumulate across these `S_k` iterations, or does each step update independently?
2. **Outer loop (across blocks):** Standard gradient accumulation across multiple blocks/microbatches before optimizer step.

**Three semantic choices, each with different distributed costs:**

| Accumulation mode | Gradient behavior | Distributed cost | Correctness |
|---|---|---|---|
| **(a) Accumulate across `S_k` steps** | Sum gradients from all `S_k` denoising iterations of one block → one optimizer update per block | `S_k` all-reduces **before** optimizer step → blocks optimizer until full denoising loop completes | Matches diffusion training (average over denoising steps) |
| **(b) Independent updates per step** | Each of `S_k` steps does separate optimizer update | `S_k`× optimizer steps + `S_k`× all-reduces per block → optimizer overhead `S_k`×-amplified | Unclear if valid (updates mid-denoising) |
| **(c) Final-step-only gradient** | Only backprop from final denoising step's loss | 1 all-reduce per block, but **discards `S_k-1` steps' supervision** | Wasteful; contradicts diffusion objective |

Standard diffusion training uses **(a)** (accumulate across timesteps/denoising steps), which means **the optimizer step is blocked on the full `S_k`-step denoising loop AND its `S_k` gradient all-reduces** — a `S_k`×-deeper critical path than AR's one-forward-one-backward.

### T-A3. Activation memory is `S_k`×-amplified — recomputation vs memory tradeoff is more brutal
AR training with activation checkpointing: store activations at layer boundaries, recompute within a checkpoint segment during backward.

dLLM training must store activations **for all `S_k` forward passes** to compute gradients, because:
- Each denoising step operates on **different masked inputs** (masks change step-to-step).
- Backward from step `t`'s loss needs step `t`'s activations, which depend on step `t`'s specific mask pattern.
- You cannot "just recompute" step `t`'s activations during backward without replaying the exact mask schedule up to step `t`.

**Memory cost:** For one block, naively storing all activations across `S_k` steps:
```
activation_memory = S_k × num_layers × bs × B × hidden
```
vs AR's `num_layers × bs × 1 × hidden` per token.

**Recomputation strategy:**
- **Selective checkpointing:** Store only certain denoising steps' activations (e.g., every `k`-th step), recompute intermediate steps during backward. Requires replaying the **mask schedule deterministically** — so mask sampling must be reproducible (seeded RNG) and synchronized across ranks.
- **Memory/compute tradeoff is harsher:** AR recomputation is cheap (one forward per token). dLLM recomputation means **re-running multi-step denoising sequences** during backward, potentially `S_k/2` forwards per backward on average if checkpointing every 2nd step.

**Distributed consequence:** Activation memory scales with `S_k`, so per-GPU memory limits batch size more severely. Under TP/FSDP, activation sharding becomes critical earlier than in AR training.

### T-A4. Bidirectional attention in backward — gradient flow is not causal
AR training: gradients flow backward through a causal mask (token `t` only affects tokens `> t`). Sparse gradient structure → can optimize gradient computation/storage.

dLLM training: bidirectional/encoder-only attention (A4) means:
- **Every token in a block has gradients from every other token in the block** (full `B × B` attention).
- Gradient tensors are **dense**, not causally sparse.
- Under TP with head-sharding, gradient all-reduce must **reconstruct full attention gradients** across ranks before applying to parameters.

**Distributed cost:**
- Gradient tensor sizes are larger (no causal sparsity).
- Gradient all-reduce volume includes full `[B, B]` attention weight gradients per layer, not upper-triangular.
- FlashAttention backward for encoder-only attention has different memory access patterns than causal — may affect kernel efficiency and thus exposed comm time during backward overlap.

### T-A5. MoE expert gradient imbalance under EP — routing drift across `S_k` steps
Inference A3 noted that expert routing drifts across denoising steps as masks resolve. In training, this causes **gradient load imbalance across expert shards**:
- Step 1 (many masks): tokens routed relatively uniformly across experts.
- Step `S_k` (few/no masks): tokens concentrate on fewer experts (e.g., high-confidence tokens use specialized experts).
- Across `S_k` backwards, **each expert accumulates gradients from a time-varying token set**, and different experts see different gradient volumes.

**Consequence under EP (expert parallelism):**
- Some ranks accumulate large expert gradients, others accumulate small gradients.
- Gradient all-reduce becomes **imbalanced** — fast ranks wait for slow ranks (straggler effect in gradient sync).
- Optimizer step timing diverges across ranks (some experts update faster).
- **Solution:** Gradient bucketing + asynchronous all-reduce, or load-balancing losses (auxiliary losses to regularize routing), but these add communication overhead or training instability.

### T-A6. Loss computation is per-step, per-position — not per-sequence
AR training loss: sum of per-token cross-entropy over the sequence, one term per output token.

dLLM training loss: typically **sum over all `S_k` denoising steps**, with each step contributing a per-masked-position cross-entropy:
```
L = Σ_{step=1}^{S_k} Σ_{pos ∈ masked_at_step} CE(logits[step, pos], target[pos])
```

**Distributed implications:**
- Loss aggregation happens **after all `S_k` steps complete** (T-A2 accumulation mode (a)).
- Cross-rank loss reduction (for logging/monitoring) must gather per-step losses, not just final loss.
- **Gradient scaling:** If accumulating across `S_k` steps, gradients must be scaled by `1/S_k` (or `1/(S_k × num_masked)`) to avoid exploding gradients — this scaling must be **rank-consistent** under TP/DP.

### T-A7. Mask scheduling is part of the training algorithm — cross-rank consensus required
Unlike AR where input tokenization is deterministic, dLLM mask scheduling can be:
- **Confidence-based** (remask low-confidence positions) — non-deterministic, depends on model outputs.
- **Random** (fixed masking ratio per step) — deterministic if seeded, but requires synchronized RNG across ranks.

**Distributed challenge:**
- Under TP, different ranks see different attention head outputs → may compute **different confidence scores** → select different positions to remask → **divergent mask schedules across ranks** → gradient corruption.
- **Solution:** Either (i) gather full logits before confidence-based masking decision (expensive all-gather per step), or (ii) use deterministic, seed-based mask scheduling independent of model outputs (simpler, but loses adaptive masking benefits).

### T-A8. Checkpointing bloat — must save mask schedules, not just parameters
AR training checkpoint: model parameters + optimizer states.

dLLM training checkpoint must additionally save:
- **Mask schedule state:** RNG seed, current step in curriculum, per-sample mask patterns if using adaptive strategies.
- **Per-step intermediate states** if using curriculum learning (e.g., gradually decreasing mask ratio, increasing `S_k`).

**Size and synchronization:**
- Checkpoint size grows with batch size if storing per-sample masks.
- Under DP/FSDP, all ranks must agree on the global mask schedule state before checkpointing → requires collective to gather/reconcile RNG state or explicit schedule broadcast.

### T-A9. Optimizer state distribution under FSDP/ZeRO — activation memory amplifies sharding pressure
Fully Sharded Data Parallel (FSDP/ZeRO) shards optimizer states + gradients + parameters across ranks. dLLM's `S_k`×-amplified activation memory (T-A3) means:
- **Less memory available for optimizer state sharding** → must use more aggressive sharding (ZeRO-3 instead of ZeRO-2) earlier.
- **Gradient sharding and all-gather overhead is `S_k`×-amplified:** Each of `S_k` backward passes must all-gather parameter shards for gradient computation → `S_k` all-gathers per block, not 1 per token.
- **Communication/computation overlap is harder:** FSDP relies on overlapping all-gather of next layer's parameters with current layer's computation. With `S_k` iterations, the overlap opportunity is smaller per iteration, and the exposed comm (T-A1) eats into compute time.

### T-A10. Multi-node training — `S_k`×-amplified cross-node gradient sync
Inter-node bandwidth is the bottleneck for large-scale training. dLLM's `S_k`× gradient communication (T-A1) multiplies this bottleneck:
- **Gradient all-reduce volume across nodes is `S_k × B`-fold larger per output token than AR.**
- **PP (pipeline parallelism) interactions:** AR training pipelines microbatches across pipeline stages, hiding inter-node communication. dLLM's phase-pure scheduling (A6 inference analog) and `S_k`-step inner loop make pipelining **harder to apply** — you cannot pipeline within a denoising loop (backward depends on forward of same step), only across blocks. So PP's bubble overhead may be higher.

| # | Distributed assumption AR training relies on | How dLLM training breaks it | dLLM penalty factor |
|---|---|---|---|
| T-A1 | Gradient comm ∝ output tokens | Gradient comm ∝ `S_k` steps × full block | ~15× gradient volume/token |
| T-A2 | One forward → one backward → accumulate → update | `S_k` forward/backward in inner loop → nested accumulation semantics | `S_k`×-deeper critical path |
| T-A3 | Activation memory ∝ batch × layers | Activation memory ∝ `S_k` × batch × layers | `S_k`× activation memory |
| T-A4 | Causal gradient flow, sparse attention grads | Bidirectional → dense gradients, full attention grad matrices | Larger grad tensors, no causal sparsity |
| T-A5 | Uniform expert gradient load | Routing drifts across `S_k` → gradient imbalance | Straggler effect in grad sync |
| T-A6 | Loss = sum over tokens | Loss = sum over `S_k` steps × masked positions | Requires per-step loss tracking, scaling |
| T-A7 | Deterministic tokenization | Mask scheduling may be confidence-based → rank-divergent | Requires consensus or deterministic seeding |
| T-A8 | Checkpoint = params + optimizer | Must also save mask schedule state, curriculum state | Checkpoint bloat |
| T-A9 | FSDP shards params/grads/optim | `S_k`×-amplified activation memory → more aggressive sharding | `S_k`× all-gather overhead |
| T-A10 | PP hides cross-node comm | `S_k` inner loop + phase-pure → harder to pipeline | Higher bubble overhead, `S_k`× cross-node grad sync |

---

## Part T-B — Additional training challenges from project algorithms (Streaming Block Controller + Selective Recompute)

The inference algorithms (Designs 1 & 2) were framed as inference-time mechanisms. If applied to training, they introduce new distributed training challenges:

### T-B1. Streaming Block Controller in training — dynamic sequence length breaks batch shape invariants
Design 1 (Streaming Block Controller) allows variable-size decoding windows across steps (B1 inference). In training:
- **Activation checkpointing must handle variable-length sequences:** Cannot preallocate fixed-size activation buffers.
- **Gradient accumulation across variable steps:** If block size changes dynamically during training (e.g., curriculum starts with small blocks, grows over training), gradient norms and learning rate schedules must adapt.
- **Batch composition:** Different samples in a batch may have different `S_k` (some blocks finish early, A5 lockstep issue). Training must either (i) pad/mask to uniform `S_k` (wasteful), or (ii) implement ragged batch backward (complex).

### T-B2. Selective Recompute in training — sparse active sets in backward pass
Design 2 (Selective Recompute) uses confidence-based active sets (only recompute uncertain positions, B2 inference). In training:
- **Which positions' gradients do you compute?** If only "active" positions are recomputed forward, their gradients flow back. But **training loss is defined over the full target sequence**, so you must either (i) compute gradients for all positions (defeating the sparsity), or (ii) accept biased gradients (only updating parameters based on uncertain positions — unclear if this converges).
- **Gradient masking:** If treating inactive positions as "frozen," their gradients must be masked/zeroed. But frozen positions were decided based on **current model outputs** (confidence), which change as training progresses → adaptive gradient masking schedule, harder to reason about convergence.
- **T-B3 mirror:** Confidence-based active set selection under TP (B3 inference) requires per-layer consensus — but now in **both forward and backward**, doubling the consensus overhead per step.

### T-B3. Draft-Token State in training — how to backprop through state transitions?
Design 2's three-state machine (`MASK → DRAFT → UNMASK`) is designed for inference. In training:
- **Loss over which states?** Do you compute loss only on `UNMASK` tokens (final outputs), or on `DRAFT` tokens too (treating draft predictions as supervision signal)?
- **Gradient flow through state transitions:** A token that transitions `MASK → DRAFT → UNMASK` across three steps participates in three forwards with different embeddings (masked embedding → draft token embedding → final token embedding). How do gradients flow back through these **in-place state mutations**?
- **Revisable cache in backward (B4 training):** If a `DRAFT` token is demoted to `MASK`, the forward pass invalidated its cached KV. During backward, do you recompute the invalidated states, or treat them as detached (no gradient through invalidated paths)?

**Challenge:** The state machine was designed for inference **correction**, not training **optimization**. Training through it requires defining a differentiable loss over state transitions, which may need auxiliary losses (e.g., draft token prediction accuracy) — adding training complexity.

### T-B4. Confidence calibration requires additional training signals
`efficient_dllm_sys.md` §6 noted confidence is unreliable without calibration. In training:
- **Need explicit confidence calibration loss** (e.g., Expected Calibration Error) on top of the base dLLM loss.
- **Calibration is rank-dependent under TP** (B5 inference) → training must either (i) calibrate per-rank (wasteful), or (ii) gather full logits for confidence computation (expensive all-gather per step in training loop).

---

## Part T-C — Training-only challenges with no inference analog

### T-C1. Curriculum learning for mask ratio and `S_k` — distributed schedule coordination
dLLM training often uses curriculum: start with high mask ratio and/or high `S_k` (easier task), gradually decrease mask ratio and/or `S_k` (harder task). 
- **Global schedule state must be synchronized across all ranks** (T-A8).
- **Gradient statistics change as curriculum progresses** → learning rate schedules, gradient clipping thresholds, and optimizer hyperparameters may need adjustment.
- **Checkpointing must save curriculum phase** to resume correctly.

### T-C2. Iterative refinement during training — how to sample denoising trajectories?
During AR training, the forward pass is deterministic (greedy or temperature-based sampling, but input is fixed once). During dLLM training:
- **Mask sampling:** Uniform random, confidence-based, or fixed schedule?
- **Remasking strategy:** Do you remask during training (to teach the model to handle iterative refinement), or use a fixed `S_k`-step schedule (simpler, but may not match inference behavior)?
- **Exposure bias:** If training uses a fixed `S_k` but inference uses adaptive refinement (confidence-based remasking), the model is trained on a different distribution than it sees at inference → potential train/test mismatch.

### T-C3. Diffusion timestep scheduling — training uses different noise schedules than inference
If the dLLM uses a diffusion-style noise schedule (adding noise to embeddings, denoising over timesteps), training typically samples timesteps uniformly or with a curriculum. Inference may use:
- **Fewer steps than training** (e.g., train with `S_k=20`, infer with `S_k=10` using DDIM-style skipping).
- **Different noise schedule** (e.g., cosine schedule in training, linear in inference).

**Distributed implication:** The `S_k` at training time determines the gradient communication cost (T-A1). If training uses high `S_k` but inference uses low `S_k`, the training cost is `S_k_train`×-amplified, even though inference benefits from fewer steps. **Training is more expensive than inference by a larger factor in dLLM than in AR** (AR training and inference have similar per-token costs; dLLM training may use `S_k=20` while inference uses `S_k=10`, so training is `20/10 = 2×` more expensive per token beyond the base amplification).

### T-C4. Token-level loss masking — not all positions contribute to loss at all steps
AR training: every output token contributes to loss. dLLM training: only **masked positions at each step** contribute to loss.
- **Loss sparsity patterns change per step:** Step 1 has many masked positions → dense loss. Step `S_k` has few/no masked positions → sparse loss.
- **Gradient magnitudes vary across steps:** Early steps have large gradients (many positions), late steps have small gradients (few positions).
- **Distributed consequence:** Gradient all-reduce volume is **not uniform across the `S_k` steps** — early steps have larger gradients, late steps have smaller. Overlap strategies must account for this non-uniformity.

---

## 3. Bottom line — training vs inference

### Training is strictly harder than inference in distributed dLLM
- **Inference challenges (Part A/B) all have backward-pass mirrors (T-A/T-B), and training adds T-C challenges (curriculum, loss masking, exposure bias) with no inference analog.**
- **Gradient communication is `S_k`×-amplified AND fully exposed in backward critical path** (T-A1), with no possibility of overlap (vs inference where overlap was only algorithmically disabled, not semantically impossible).
- **Activation memory is `S_k`×-amplified** (T-A3), forcing aggressive FSDP/ZeRO and making large-batch training harder.
- **Mask scheduling must be rank-consistent** (T-A7), requiring either expensive consensus or deterministic seeding (losing adaptive masking).
- **Checkpointing is larger and more complex** (T-A8), must save mask schedules and curriculum state.

### Multi-node training is the critical bottleneck
- **Gradient all-reduce across nodes is `S_k × B`-fold larger per output token** (T-A10).
- **PP is harder to apply** due to phase-pure scheduling and `S_k` inner loop.
- **Inference's "multi-node may not be viable" warning (A8) becomes a training certainty:** Without step reduction or better gradient overlap, multi-node dLLM training may be **impractical at scale**, forcing training to large single-node systems (8×H100/A100 NVLink pods) rather than multi-node clusters.

### Project algorithms (Designs 1 & 2) are **inference-optimized, not training-friendly**
- Streaming Block Controller's dynamic shapes (T-B1) complicate activation checkpointing and batch backward.
- Selective Recompute's sparse active sets (T-B2) conflict with full-sequence loss computation — unclear if biased gradients converge.
- Draft-Token State's in-place mutations (T-B3) require defining differentiable loss over state transitions, adding training complexity.

**Implication:** The project's research algorithms are likely **inference-only contributions**. If pursuing training, the base dLLM challenges (T-A) must be solved first (better gradient overlap, deterministic mask scheduling, activation memory reduction) before considering algorithmic extensions.

### Open questions for distributed dLLM training research
1. **Can gradient communication be overlapped with backward compute under the `S_k`-step inner loop, or is the critical path intrinsically `S_k`×-deeper?** (T-A1)
2. **What is the right gradient accumulation semantic for diffusion training — accumulate across `S_k` steps (mode a) or treat each step independently?** (T-A2)
3. **Can activation checkpointing be done more aggressively using deterministic mask replay, or does stochastic mask sampling force full activation storage?** (T-A3)
4. **Does FSDP/ZeRO stage-3 sharding + `S_k`× activation memory make dLLM training fundamentally memory-bound, or can mixed-precision/quantization help?** (T-A9)
5. **Is multi-node dLLM training viable without reducing `S_k` below ~5 steps, or must training be confined to large single-node systems?** (T-A10)
6. **Can the project's inference algorithms (Designs 1 & 2) be adapted to training with well-defined gradient semantics, or should they remain inference-only?** (T-B)
7. **How severe is exposure bias (train with fixed `S_k`, infer with adaptive refinement), and does it require explicit train-time refinement simulation?** (T-C2)

### Next steps
- **Measure training-specific distributed costs:** Extend D2 profiling to training mode — track per-step gradient all-reduce volume, activation memory usage, backward pass time breakdown, FSDP all-gather overhead.
- **Validate gradient accumulation semantics:** Implement modes (a), (b), (c) from T-A2, measure convergence and communication cost on a small model/dataset.
- **Test mask scheduling strategies under TP:** Measure whether deterministic seeding vs confidence-based masking affects (i) training convergence, (ii) rank divergence risk, (iii) gradient synchronization cost.
- **Establish single-node baseline before multi-node:** Focus on 8×A100/H100 single-node training first (where NVLink bandwidth is high). Only attempt multi-node if single-node results show `S_k` can be reduced to ~5 or gradient overlap is effective.