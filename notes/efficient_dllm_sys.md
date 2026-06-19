# Efficient dLLM System — Project Explainer

> **Working title:** *A State-Aware Execution System for Efficient dLLM Inference*
>
> This document expands the draft slide deck (`dllmsys.pdf`) into a structured explanation of the project's motivation, challenges, methodology, and current progress. The deck is an early draft and its slides are out of order; the narrative below reorganizes them into a coherent flow. Where the slides are terse or ambiguous, interpretation is flagged explicitly with *(interpretation)*.

---

## 1. One-paragraph summary

Diffusion LLMs (dLLMs) promise faster generation than autoregressive (AR) models by decoding many tokens in parallel, but in practice modern block-wise dLLMs (e.g., LLaDA-2.0) can cost **up to ~8× more compute than AR decoding** to produce the same output. 

The project's thesis is that **this waste comes from treating a fixed block as the atomic execution unit**, which couples two concerns that should be decoupled: *how much decoding progress is made per step* and *how much of the network is recomputed per step*. 

The proposed system is an **algorithm-level, state-aware execution layer** that 

- (1) **dynamically sizes the decoding window** (*Streaming Block Controller*) to make fewer, more productive steps, and 
- (2) **selectively recomputes only the tokens/layers whose state actually changed** (*Selective Recompute / Draft-token state machine*) to lower per-step cost — all **without writing new GPU kernels**.



---

## 2. Background

### 2.1 The dLLM decoding paradigm

A masked diffusion LLM (e.g., **LLaDA**) generates by iterative denoising rather than left-to-right token emission:

1. Initialize the response region as all `MASK` tokens, appended to the prompt.
2. At each step, run a **full-attention** forward pass over the *entire* sequence and predict a distribution for every masked position.
3. Score each masked position's top prediction by confidence.
4. **Unmask** the top-k most confident positions (commit those tokens); leave the rest masked.
5. Repeat until all positions are unmasked.

The reference loop from the deck (LLaDA):

```python
# init: response region is all MASK
x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long)
x[:, :prompt_len] = prompt

for step in range(num_steps):
    mask_index = (x == MASK_ID)
    logits = model(x).logits                 # full-attention over all positions
    x0 = torch.argmax(logits, dim=-1)         # predict every position
    probs = F.softmax(logits, dim=-1)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, confidence, -float('inf'))
    k = gen_length // steps                   # fixed unmask budget per step
    _, topk_indices = torch.topk(confidence[0], k=k)
    x[0, topk_indices] = x0[0, topk_indices]  # commit only the most confident
return x
```

The key difference from AR: every step recomputes attention over the **whole** sequence (masked + unmasked), and the model is **bidirectional** rather than causal.

### 2.2 Block-wise dLLMs (the modern paradigm)

Pure full-sequence diffusion is expensive and cannot reuse a KV cache, so current high-quality dLLMs use a **block-wise / semi-autoregressive** execution scheme. **SDAR** ("Synergistic Diffusion-AutoRegression") and **LLaDA-2.0** are the representative examples the deck builds on:

- **Inter-block: autoregressive.** The response is split into fixed-size blocks. 
  - Blocks are generated left-to-right under a block-wise *causal* attention mask, so completed blocks can be cached in a **KV cache** and reused.
- **Intra-block: diffusion.** 
  - Within the current block, tokens are decoded in parallel by iterative masked denoising. The block currently being decoded does *not* store a KV cache (its content is still changing).

This hybrid keeps the parallelism benefit inside a block while recovering the KV-cache reuse benefit across blocks. The project also references **Delayed Cache (dKV-Cache, NeurIPS'25)** as a related "delayed cache" workflow.

### 2.3 Why dLLM decoding is expensive — the cost model

The deck formalizes the cost gap. Symbols:

| Symbol  | Meaning                                      |
| ------- | -------------------------------------------- |
| `L`     | output length                                |
| `B`     | block size                                   |
| `K`     | number of blocks, `K = L / B`                |
| `C_AR`  | cost of one AR decoding step                 |
| `C_dlm` | cost of one dLLM decoding step for a block   |
| `S_k`   | number of diffusion steps spent on block `k` |
| `N̄`     | average tokens committed per step            |

- **AR decoding:**  $T_{AR} ≈ L × C_{AR}$
- **dLLM decoding:**  `T_dlm ≈ Σ_{k=1..K} S_k · C_dlm ≈ K · S̄ · C̄ ≈ (L / N̄) · C̄`

The intuition: dLLM total cost scales with the **number of steps** (`L / N̄`) times the **cost per step** (`C̄`). In current block-wise dLLMs (LLaDA-2.0), end-to-end decoding can be **up to ~8× the cost of AR decoding** for the same output.

This yields **two orthogonal optimization targets**, which structure the entire project:

1. **Fewer steps** → raise `N̄` (decode more tokens per step).
2. **Lower step cost** → shrink `C̄` (reduce effective work per step).



### 2.4 What dLLM decoding actually looks like (analysis)

Empirical observations about *how* dLLMs decode, which constrain the design space:

- The overall generation trend **remains essentially autoregressive** (tokens tend to resolve left-to-right).
- **Parallel decoding is limited to mid-/short-range regions** — long-range parallelism rarely materializes.
- Within a decodable range, the model naturally **resolves easy tokens first, then hard tokens**.
- **Output length must be decided before subtasks can be parallelized.** Because positional encoding imposes a strict positional dependency, you cannot parallelize multiple output "subtasks" until you know where they sit. *(interpretation: this is why simply widening the parallel window is not free.)*





---

## 3. Motivation

The slides motivate the two optimization targets with empirical evidence, then collapse them to a single root cause.

### 3.1 Opportunity A — Fewer steps: token convergence within a block is uneven

Measured on **LLaDA2.0-mini, GSM8K** (32 prompts, 311 blocks): **the average number of tokens decoded *per step within a block* is highly front-loaded.**

- **Early steps decode many easy tokens** (the first step alone commits ~10+ tokens on average).
- **Later steps make very little progress** (often ~1 token/step), yet still pay a full forward pass.

The culprit is **fixed block boundaries**: once a block is "mostly done," the remaining stubborn tokens force many cheap-progress steps before the block closes and the next one opens. 

**Breaking the rigid block boundary** would let the system keep committing easy tokens that lie *beyond* the current block instead of grinding through a block's tail.



### 3.2 Opportunity B — Lower step cost: cross-step redundancy

Across consecutive diffusion steps, most of the computation is redundant. The deck identifies redundancy at **three levels**:

1. **Input embedding.** Between steps, the input sequence changes only at the few positions that were newly decoded; **the vast majority of input embeddings are identical step-to-step.**
2. **Intermediate state.** Per-layer hidden/attention states (visualized as Q cosine-similarity heatmaps across layers, "green = stable, red = changing") remain highly similar across many positions and steps.
   - Crucially, **cross-step changes propagate only gradually across layers** — a change at the input diffuses outward layer by layer rather than perturbing everything at once.
3. **Decoding output.** *(Listed but struck through / deprioritized in the deck — see token categories in §5.2.)*

Because full-block recomputation **treats every token and every layer equally**, it wastes work re-deriving states that did not change. **Only the changed inputs and the intermediate states they actually affect need to be recomputed.**



### 3.3 Root cause

Both inefficiencies trace to a single design choice:

> **The full block is used as the execution unit** — for *both* decode progress *and* state update.

- Using the block as the **progress unit** → fixed boundaries cap how many tokens can be committed per step (Opportunity A).
- Using the block as the **state-update unit** → every step recomputes the whole block even though state changes are sparse (Opportunity B).

The project's response is to **decouple these**: a flexible *decoding window* for progress, and a *fine-grained state machine* for recompute.



---

## 4. Key Observations (the evidence base)

Three observations (measured mostly on the **SDAR** baseline with **HumanEval**) ground the designs.

### Observation 1 — Decoding efficiency is highly uneven

Convergence speed within a block is very non-uniform: **high decoding efficiency in the early stage of a block, low efficiency in the later stage** (consistent with §3.1). Motivates the **Streaming Block Controller** (Design 1).

### Observation 2 — Delayed unmask (tokens are correct long before they're committed)

Token state is **binary** today: a position is either `MASK` or `UNMASK`. Comparing step 1 vs. step 10 of a block on HumanEval, **many tokens are already correct at step 1**, but the model only lets their **confidence "pass" the threshold at step ~10**, and **recomputes them repeatedly** in between. This is pure wasted work and motivates an intermediate **"draft" token state** (Design 2).

### Observation 3 — Unmasked tokens were already high-ranked the step before

Tokens selected for unmasking in the current step **usually already had a high rank in the previous step** (confidence-vs-diffusion-step trajectories climb steadily, then cross threshold). This means upcoming commits are **predictable from prior state**, which is what makes selective recompute and draft-promotion safe. *(interpretation: predictability is the safety argument behind freezing most tokens.)*



---

## 5. Methodology / System Design

The system has two complementary mechanisms, one per optimization target.

### 5.1 Design 1 — Streaming Block Controller (target: fewer steps)

**Idea:** dynamically adjust the **decoding window** (a "streaming block") instead of using a fixed block, **expanding the window only when the gain is worth the cost.**

The trade-off being balanced:

- **Fixed blocks** → limit decoding progress (Opportunity A); window too small.
- **Fixed masked-token constraint** (e.g., "always keep N masked slots open") → can **over-expand** the window (the deck cites growth from **32 → 80+** positions).
- **Windows that are too large increase hardware overhead** (more positions to attend over per step).

So the controller **dynamically chooses a window size that balances decoding progress against per-step hardware cost** — a hardware-aware streaming strategy that operates under a "HW decoding budget." 

- Visually: instead of a fixed block marching down the diagonal, the active decoding region streams and resizes to track where productive unmasking is actually happening.

**Effect on the decode profile:** under streaming, the per-step decoded-token curve on LLaDA2.0-mini/GSM8K flattens out (sustained ~3–6 tokens/step across many more in-block steps, with fewer total blocks: 311 → 289), instead of collapsing to ~1 token/step in the block tail.

**Preliminary latency results** (Streaming controller; baseline `BL`):

| Prompt | BL Latency | +Streaming (A32) | +Streaming (2-Block) |
| ------ | ---------- | ---------------- | -------------------- |
| code   | 7.27s      | 6.54s (1.11×)    | 5.56s (1.31×)        |
| math1  | 7.30s      | 6.53s (1.12×)    | 6.48s (1.13×)        |
| math2  | 4.40s      | 2.29s (1.92×)    | 2.65s (1.66×)        |
| text   | 18.00s     | 17.14s (1.05×)   | 18.11s (0.99×)       |

Speedups are real but **workload-dependent** (strong on math, marginal/none on free-form text), and every row carries a ⚠️ in the deck — *(interpretation: these are early numbers pending correctness/quality verification, not final results).*



### 5.2 Design 2 — Draft Token State + Selective Recompute (target: lower step cost)

**Idea:** replace the binary mask/unmask state with a **three-state token machine**, and recompute only what changed.

**Token states:**

| State      | Compute behavior                                             |
| ---------- | ------------------------------------------------------------ |
| **MASK**   | Full calculation every step.                                 |
| **DRAFT**  | **Soft cache** — tentatively decoded; supports update/edit; cheap to revise. |
| **UNMASK** | **Full cache** — committed; **skip calculation and update** (frozen). |

**State transitions** (governed by confidence thresholds `τ_low`, `τ_high`):

- `MASK → DRAFT` when confidence enters the band `(τ_low, τ_high)`.
- `MASK → MASK` (stay) — *(a high-confidence prediction may be committed directly; the diagram shows a `confidence > τ_high` self-loop on MASK)*.
- `DRAFT → MASK` (demote) when **confidence drops sharply or content changes abruptly**.
- `DRAFT → UNMASK` (**irreversible**, "不可逆") when **confidence > `τ_high` AND neighbors are stable AND the token is unchanged for `N` consecutive steps.**

This directly attacks Observation 2: tokens that are "correct early" sit cheaply in DRAFT (revisable, lightly computed) and get promoted to frozen UNMASK only once they're robustly stable — instead of being fully recomputed every step until their confidence crosses a threshold.

**Selective Recompute** (the compute mechanism that exploits §3.2's gradual cross-layer propagation):

1. Start from the tokens **changed in the current step** as the **active Q** set.
2. **Freeze all other tokens as KV-only context** (they contribute keys/values but aren't re-projected as queries).
3. **Update only the attention rows/columns touched by the active Q.**
4. **Re-activate only the KV tokens significantly affected** by these changes (detected by comparing fresh partial attention scores against cached scores), propagating the active set to the next layer.

In other words, a sparse set of changed tokens drives a sparse, *growing* recompute front that expands layer-by-layer only as far as influence actually reaches — rather than a dense full-block recompute. Attention-score-gradient heatmaps show the affected KV positions are indeed sparse per step.





---

## 6. Challenges

The hard problems, mostly concentrated in Design 2 (verifying that selective recompute and draft-freezing don't corrupt output):

- **Verification under bidirectional attention.** dLLMs attend both directions, so a committed token's correctness depends on *future* context that may still change.
- **One token can trigger global attention updates.** A single newly decoded token can, in principle, perturb attention everywhere — so "only changed tokens matter" is a useful heuristic, not a guarantee.
- **Confidence from draft/unmasked inputs is not reliable.** Once tokens are cached as draft/unmask, the confidence the model reports for downstream positions is computed against partially-frozen context and **cannot be trusted at face value** for accept/reject decisions.

**Open TODOs** to address these:

- **Offline + runtime analysis** to estimate *reliable* decoding confidence (rather than trusting raw per-step confidence).
- **Accept tokens only when multiple decoding steps agree** on the same result (temporal consistency as a correctness proxy — connects to Observation 3).
- **Remask strategies** to maintain generation correctness — i.e., the ability to pull a token back from DRAFT to MASK when later context contradicts it.



---

## 7. Current Progress & Evaluation Results

### 7.1 Threshold sensitivity (`m` = mask threshold, `u` = unmask threshold)

Sweeping the two thresholds on **LLaDA2-mini (20L)** and **SDAR-8B** Dynamic Cache:

- **Headline finding: accuracy is much more sensitive to `m` (mask threshold) than to `u` (unmask threshold).** The unmask threshold can be loosened to gain sparsity at little accuracy cost; the mask threshold must be tuned carefully.
- On the **SDAR-8B** score-vs-sparsity sweep (`m` swept; baseline full Dynamic Cache = 72.56 HumanEval pass@1): small `m` (≤0.03) stays near baseline while gaining sparsity, but accuracy **collapses sharply** as `m` grows (e.g., `m=0.08` → ~41, `m=0.1` → ~19, `m=0.15` → ~2).
- On the **LLaDA2-mini HumanEval grid** (baseline pass@1 = 79.3): many `(m, u)` settings **match or beat** baseline while cutting density — e.g., `m=0.04, u=0.1` reaches **84.2 (+4.9)** at sparsity ~0.95; several configs land in the +1 to +4 range. Sparsity here = `1 − avg(Q_set)/block_length`.



### 7.2 Sparsity / density analysis

- **Density** is defined as the **average active-token ratio** (fraction of tokens kept in the active-Q set rather than frozen as KV-only).
- Per-layer sparsity curves show the **cross-layer propagation pattern** from §3.2: the active-token ratio is **lowest in early layers** (changes are still local) and **grows with depth** as influence propagates — confirming that freezing early-layer tokens is where most of the savings come from, and that loosening `u` raises per-layer density roughly monotonically.



### 7.3 End-to-end evaluation

Across **GSM8K, HumanEval, IFEval, MATH-500, MBPP** on two backbones (SDAR-8B and LLaDA2-mini), comparing **Baseline (LLaDA-2.0, block=32)**, **Delayed Cache (dKV-Cache, NeurIPS'25)**, **FOCUS**, and the proposed **DynamicCache (`u` sweep)**:

- **The proposed method outperforms both the LLaDA-2.0 baseline and Delayed Cache** — it traces a favorable accuracy-vs-sparsity frontier (sweeping `u` keeps accuracy near/above baseline while reaching meaningful sparsity).
- **FOCUS reaches lower density (higher sparsity)** than the proposed method, **but severely hurts accuracy in some cases** — i.e., it pushes sparsity too aggressively and falls off the accuracy frontier on certain tasks.

*(interpretation: the project positions itself as the better accuracy/sparsity trade-off — not the absolute sparsest, but the one that holds quality.)*



### 7.4 Related work positioned in the deck

- **FOCUS (arXiv 2026.02).** Training-free dLLM acceleration. 
  - Method: unmasked tokens go to **direct cache**; masked tokens use **early-exit based on a delta-attention value at Layer 1** ("token eviction" — full compute through Layer-1 Q/K projections, rank decodable candidates by an *importance delta*, keep only the top candidates within an **adaptive token budget**). 
  - It's the most aggressive sparsity baseline and the main accuracy-trade-off foil (see §7.3).
- **Delayed Cache (dKV-Cache, NeurIPS'25).** The "delayed cache" workflow the design generalizes; used as a direct E2E baseline.

---

## 8. Current Status & Concerns (self-assessment)

From the deck's own summary:

- **Motivation:** there are higher parallel-decoding opportunities latent in dLLM inference than current block-wise schemes exploit.
- **Current exploration:** *adaptive decoding strategies* — a **dynamic decoding window** (Design 1) and **dynamic unmasking policies** (Design 2) — driven by **offline analysis + runtime adaptation**.
- **Current concern (the honest gap):** the work is **mostly at the algorithm level; system-level design is still limited/lacking.** Specifically:
  - The design **introduces no new GPU kernels.**
  - The decoding strategy is **decided before each step** (no dynamic behavior *within* a step).
  - **Existing kernels can be reused** as-is.

*(interpretation: this is framed as both a strength — easy to deploy on existing runtimes — and a weakness for a systems venue, which would want kernel-level contributions. A likely next step is pushing the state-aware logic down into custom attention kernels so the "selective recompute" sparsity translates into real wall-clock speedups rather than just FLOP/density reductions.)*

---

## 9. Open questions / suggested next steps

*(These are not in the deck — offered as a reading aid / discussion prompts.)*

1. **Density ≠ latency.** §7 reports sparsity/density wins, but §5.1 latency wins are modest and workload-dependent. Closing the gap between *theoretical sparsity* and *measured speedup* (likely via kernels, per §8) seems like the central systems risk.
2. **Correctness guarantees for irreversible UNMASK.** Given bidirectional attention, under what conditions is freezing provably safe? The "N steps unchanged + stable neighbors" rule is a heuristic — quantifying its failure rate per task would strengthen the claim.
3. **Interaction between the two designs.** Streaming windows (Design 1) change which tokens are "in flight," which changes the change-set that drives Selective Recompute (Design 2). Are they evaluated jointly or only in isolation?
4. **Threshold transfer.** `m`/`u` are tuned per backbone (SDAR-8B vs. LLaDA2-mini). Do good settings transfer across models/tasks, or is per-deployment tuning needed?

```

```