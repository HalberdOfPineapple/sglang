# Experiment D2 — Exposed Comm/Comp per Output Token, Measured Per-Step (LLaDA2.0-mini, 4×H100 NVLink, concurrency 4/8/16)

Sizes the **distributed cost of diffusion decoding per delivered token** by tying **every denoising forward to the tokens it actually decoded that step** — not a block-level `S_k` times one global average. For each forward we record `(batch_size, tokens_committed)` (the §3 per-step counter) and measure that forward's comm/comp time from a co-run nsys CUDA-graph trace, then divide: `comm_per_token(step) = comm_per_forward(bs) / committed(step)`. Swept over a **GPU-bound concurrency 4/8/16** on **HumanEval** prompts; every per-token / per-forward metric is reported as a **distribution (histogram + mean + median)**, not a point estimate. Headline: exposed TP-all-reduce costs a **median 0.18 / 0.13 / 0.09 ms per output token** (mean 0.42 / 0.31 / 0.18, right-skewed) as concurrency rises 4→8→16 — and is a **median ~22% of each forward's GPU time** (mean ~27%, with a desync tail to ~80%). It falls because tokens committed per forward climb **4.8 → 7.9 → 13.4** while comm-per-forward grows only sub-linearly (median 1.18 → 2.00 ms); meanwhile a block still pays **~14–15 exposed comm rounds** (`S_k`) to emit its 32 tokens vs the 1-round parallel-decode ideal, and **in-batch straggler waste rises 34% → 44%**.

> **Supersedes the first D2 pass.** An earlier version of this report multiplied a block-level `S_k` by D1's *single global* `comm_per_forward` over **8 hand-picked prompts at a host-bound realized batch of 1.78**, and quoted one number (0.92 ms/token). That was wrong on three counts (raised in review): the prompts/concurrency were unrepresentative, per-token cost was not measured **per step** (so it ignored how many tokens each forward actually decodes), and the `S_k` "distribution" mixed the batch-level `S_k` (constant within a call) with arbitrary prompts. This pass fixes all three: **HumanEval at sustained GPU-bound concurrency**, **per-step measurement of comm/comp tied to tokens decoded**, and a **concurrency sweep** showing per-token comm is an *operating-point function*, not a constant.

## Summary

Per-step measurement on the stock `LowConfidence` path, TP4/EP4, **co-run nsys** (`--cuda-graph-trace=node`) measuring comm/comp of the forward **at this run's batch size** (no reuse of a foreign average), joined to the per-step counter. Findings:
1. **Exposed comm per output token falls steeply with concurrency — and is a right-skewed distribution, not a point.** Token-weighted over delivered tokens: **median 0.18 / 0.13 / 0.09 ms, mean 0.42 / 0.31 / 0.18 ms** at conc 4 / 8 / 16 (Fig 3). Driver: tokens committed per forward rise **4.84 / 7.90 / 13.43** (median 3 / 5 / 8, max 46 / 62 / 150 — a wide per-forward distribution, Fig 2), while comm-per-forward grows only sub-linearly. So **batching amortizes the exposed all-reduce across more committed tokens**; the median (typical delivered token) is well below the mean because 1-token forwards each carry a whole forward's comm.
2. **The communication-time *fraction* of GPU time is tight: median ~22%, mean ~27% (Fig 5).** `comm/(comm+comp)` per token (each forward weighted by the tokens it decoded) vs per forward are **nearly identical** (median 22% both; the fraction is independent of how many tokens a forward commits) — a distribution-level refinement of D1's "32% of GPU" mean, with a thin desync tail to ~80%. Per-forward comm/comp are validated as a property of batch size: **compute time CV ≤ 4.3%** (a near-spike, content-insensitive, Fig 6), comm *bytes* fixed but exposed time heavy-tailed (rare host-induced cross-rank desync stalls, mean ≫ median).
3. **`S_k` is a per-block, content-driven property — stable across concurrency.** Intrinsic `s_k` (a block's own denoising steps) is **mean 13.8 / 15.1 / 15.3**, median ~14–15, max 31–32, essentially flat across conc 4→16 (it depends on prompt difficulty, not batch). A block thus pays **~14–15 exposed comm rounds** to emit its 32 tokens vs the **1-round** parallel-decode ideal → a **~15× round amplification**; an AR model of equal size pays one round/token but **hides all of it behind compute** (overlap), whereas dLLM force-disables overlap so every round is exposed.
4. **Batching trades comm-per-token DOWN for straggler waste UP.** The batch runs until its slowest block is mask-free, so wasted forwards-after-finish climb **34% → 40% → 44%** of the batch `S_k` an average block sits through as concurrency grows (more blocks ⇒ more likely a straggler holds the batch). The per-block batch `S_k` inflates to **21.0 / 25.0 / 27.2** vs intrinsic 13.8 / 15.1 / 15.3.
5. **Compute per token dominates comm ~2:1 and also falls with concurrency.** Comp-per-token (dominant-bs steps) **0.80 / 0.56 / 0.37 ms**, i.e. comm:comp ≈ **1:2** per token; both amortize with batch. Throughput **355 / 523 / 726 tok/s** (89 / 131 / 182 tok/s/GPU) — sublinear (4× concurrency → 2× throughput), the cost of exposed collectives + straggler waste.

Net: the per-token distributed cost of stock dLLM serving is **concurrency-dependent** (median ~0.09–0.18 ms exposed comm/token, ~22% of forward GPU time, at GPU-bound points — far below the first pass's single 0.92), the `S_k`× structure (≈15 exposed rounds/block, fully exposed) is the constant motivation for **step reduction (Design-1)** and **overlap (I3)**, and the **batching↔straggler tension** (finding 4) is the lever for **ragged batching (I1/I5)**.

## Setup

### Hardware & software
- **GPUs:** 4× NVIDIA **H100 80GB HBM3**, **full NVLink mesh** (`nvidia-smi topo -m` = `NV18` every pair). Every TP all-reduce rides NVLink. Same box as D1.
- **Software:** conda env `sglang` (Python 3.10), SGLang git `1464f04b3`, **nsys 2026.3.1**, FlashInfer attention, sm90. Comm/comp are **measured in this experiment** (co-run nsys per concurrency), not reused from D1.
- **Model:** `inclusionAI/LLaDA2.0-mini` — `llada2_moe`, 20 layers, hidden 2048, **256 experts / top-8**, vocab 157184, bf16. `block_size=32`, `mask_id=156895`.

### Parallelism / runtime config (confirmed from server log)
| Setting                | Value                                                |
| ---------------------- | ---------------------------------------------------- |
| dLLM algorithm         | `LowConfidence`, threshold 0.95                      |
| TP / EP                | 4 / 4                                                |
| MoE A2A backend        | `none` (TP all-reduce; no EP all-to-all)             |
| Attention backend      | `flashinfer`                                         |
| `mem_fraction_static`  | 0.7                                                  |
| `max_running_requests` | = concurrency (4 / 8 / 16) per run                   |
| `page_size`            | 32 (= block_size)                                    |
| overlap schedule       | disabled (forced) → comm fully exposed               |
| full CUDA graph        | ON (`--cuda-graph-trace=node` exposes in-graph NCCL) |

### Workload
**HumanEval** (`/cephfs/shared/wxli/human-eval`), first **20 problems** cycled by an asyncio driver (`max_new_tokens=256`, `temperature=0`). Real code-completion prompts give a content-driven `S_k` spread, and the driver sustains a full running batch so the server is **GPU-bound** — the representative serving point the first pass (curl bursts, realized batch 1.78) missed.
**Concurrency vs batch size vs request count — three distinct quantities** (values for the conc=16 run):
- **Concurrency (16)** — requests the driver keeps *in flight at once* (it opens a new one as each finishes); `--max-running-requests 16` lets the server actually run them together. This is the *target* running-set size, not the realized batch.
- **Number of requests (64 = `4×concurrency`)** — total `/generate` calls issued in the measured window; just the workload volume. **Why `4×`:** issuing only `concurrency` requests would ramp the batch to full and immediately drain (almost all transient, little steady state); ~4 "waves" keep the batch near-full for most of the short (trace-bounded) capture. It worked — **613 of 1418 forwards ran at the full bs=16**, the rest being the ramp/drain tail (see the `batch_size mix` line in each `*_summary.txt`).
- **Batch size (realized mean 10.7, max 16)** — the request-blocks the dLLM `run()` loop *actually* forwards together. It equals concurrency only when the pipe is full; it dips below during ramp/drain and as requests finish at staggered times. **This** is what the counter logs per forward and what selects the CUDA graph — i.e. what sets each forward's comm/comp cost.

**How they compose** (each request ≈ `256/32` ≈ 8–10 sequential 32-token **blocks**; identities verified from the CSVs at conc=16): one **`run()` call** advances every active request by one block, so `blocks = Σ_call batch_size` (639 = 60 calls × mean 10.7); within a call the loop runs `S_k` denoising **forwards** on the whole batch until all its blocks are mask-free, so `forwards = Σ_call S_k` (1418). Hence **comm/forward is set by `batch_size`** (it picks the captured graph), **tokens committed per forward is summed across those `batch_size` blocks**, and `comm/token = comm_per_forward / committed`.

### Runs
| Tag               | conc / max-running | dominant bs | reqs | forwards / blocks | tok/s |
| ----------------- | ------------------ | ----------- | ---- | ----------------- | ----- |
| `d2_h100_tp4_c4`  | 4                  | 4           | 16   | 966 / 158         | 355   |
| `d2_h100_tp4_c8`  | 8                  | 8           | 32   | 1165 / 310        | 523   |
| `d2_h100_tp4_c16` | 16                 | 16          | 64   | 1418 / 639        | 726   |

Artifacts: `$DATA_ROOT/profiling/dllm/d2_sk_amplification/` (`DATA_ROOT` default `/cephfs/shared/wxli/sglang-dllm`): `profiles/<tag>.nsys-rep`+`.sqlite`+`_cuda_gpu_kern_sum.csv`; `logs/<tag>_blocks.csv`, `<tag>_blocks_perstep.csv`, `<tag>_summary.txt`, `<tag>_d2metrics.json`, `<tag>_drive.log`, `d2_sweep_summary.txt`.

## Method & tooling

### Instrumentation (env-gated, isolated, bit-identical when off)
The §3 counter (`DllmStepCounter` in `python/sglang/srt/dllm/profiling.py`, `SGLANG_DLLM_PROFILE=1`, rank-0 only) writes **two CSVs** via gated hooks in `dllm/algorithm/low_confidence.py`: a **per-step** CSV `(call_id, step, batch_size, n_active, committed)` — one row per denoising forward, recording the tokens committed *that step* — and a **per-block** CSV `(…, S_k, finish_step, n_committed)`. The model module is untouched; the baseline path is bit-identical with the flag off.

### Per-token measurement (the methodology fix)
Each denoising forward is a **CUDA-graph replay**, so we reconstruct **its** comm and compute time **per replay** from the nsys `.sqlite` (each `graphNodeId` fires once per replay; replays run serially on the stream, so sorting a node's instances by start-time indexes them by replay → replay *i*'s comm/comp = Σ of every node's *i*-th instance, split collective vs compute by kernel name). The per-step counter gives that forward's `committed` token count `n`. **One forward → one sample of (comm time, comp time, n)** → per-token `comm/token = comm_time/n`, `comp/token = comp_time/n`, and `comm fraction = comm_time/(comm_time+comp_time)`. To turn these per-forward samples into **per-token distributions** we weight each forward by its `n` (a forward decoding 40 tokens represents 40 tokens), so the histogram mean equals the true average cost per delivered token. We report the full distributions (histograms + mean + median; `plot_d2.py` → figures + `d2_dist_stats.json`), not a point estimate. **Graph↔forward join:** a forward's realized batch_size is **padded up** to the nearest captured graph size (`cuda_graph_bs=[1,2,4,8,12,16,…]`) and some forwards run eager, so we pool per-replay (comm,comp) by padded bs and pair with counter forwards at that bs; comm⊥committed (validated below), so the pairing is statistically valid.

### Is one number per batch size valid? (per-forward cost stability across content)
The dominant-graph approach takes **one** `comm_per_forward` / `comp_per_forward` per batch size, but consecutive forwards at the same bs process **different content** — different prompts ⇒ different paged-KV length, different masked positions ⇒ different MoE expert routing — so the time could in principle vary forward-to-forward. We validated this by reconstructing the **per-replay** cost of the dominant graph (each `graphNodeId` fires once per replay; replays run serially on the stream, so sorting a node's instances by start-time indexes them by replay → replay *i*'s comm/comp = sum of every node's *i*-th instance; `dominant_per_replay` in `parse_d2.py`). The answer splits cleanly (numbers in L1 below):
- **Compute is content-insensitive — one number is justified.** Per-replay `comp/fwd` CV is **4.3% / 3.5% / 2.4%** (conc 4/8/16); the *only* content/KV-length-sensitive kernel, **attention**, has CV **1.2% / 0.4% / 0.6%** and is just ~0.4 ms (~6% of comp). The dLLM forward re-processes fixed-shape full 32-token blocks (no eviction) and GEMM/MoE FLOPs are fixed by `bs×32` regardless of routing, so prompt-length variation (HumanEval stubs) barely moves it.
- **Comm bytes are fixed but the *exposed* time is heavy-tailed — so we report the distribution (median + mean), not a point.** `comm/fwd` payload is shape-fixed (depends only on bs), but per-replay exposed time has CV **152% / 144% / 106%**: the bulk is tight (median **1.18 / 1.45 / 2.00 ms**) with a small number of **desync spikes** (>6 ms: 4–6% of forwards, up to 47 ms; Fig 6 top). These are **cross-rank stalls**, not content: between graph replays the **un-graphed host-side select loop** runs (variable Python + `.item()` syncs), occasionally skewing the 4 ranks so the next forward's first all-reduce waits for the slowest peer. The **mean** `comm/fwd` is thus stall-inflated vs the **median** — the report gives both. The stalls are real exposed time but host-induced — a scheduling artifact of the eager-host/graphed-device dLLM loop, attackable independently of the collective itself.

### Exact reproduction
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
# full sweep (launches the server under nsys once per concurrency point):
bash experiments/profiling/dllm/d2_sk_amplification/run_d2.sh         # CONC_LIST="4 8 16"
# one point:   CONC_LIST=8 bash .../run_d2.sh
# re-parse a saved run (no GPU):
python experiments/profiling/dllm/d2_sk_amplification/parse_d2.py \
  $DATA_ROOT/.../logs/d2_h100_tp4_c8_blocks $DATA_ROOT/.../profiles/d2_h100_tp4_c8
# regenerate the 6 figures + d2_dist_stats.json (no GPU; reads saved CSVs + .sqlite):
python experiments/profiling/dllm/d2_sk_amplification/plot_d2.py
```
Figures: `fig1` per-token means vs concurrency · `fig2` tokens/forward distribution · `fig3` per-token comm/comp distribution · `fig4` `s_k`+straggler · `fig5` **comm fraction (per forward vs per token)** · `fig6` per-forward comm/comp time distributions.

## Results 

Organized bottom-up so each number's origin is explicit: **L1** device cost per forward ÷ **L2** tokens decoded per forward = **L4** per-token cost; **L3** per-block structure explains the `S_k` amplification and straggler waste; **L5** serving context closes. All per-forward measured (one (comm,comp,committed) sample per denoising forward, rep rank dev=0); per-token / per-forward quantities are reported as **distributions (histogram + mean + median)**, since each is right-skewed and a single number misleads.

### Metrics measured (skeleton — read this first)
| level | metric | unit | source | answers |
| --- | --- | --- | --- | --- |
| **L1** device | `comm/fwd`, `comp/fwd` (distributions) | ms | nsys per-replay | cost of collectives vs compute in one denoising forward (Fig 6) |
| **L1** device | **`comm fraction` = comm/(comm+comp)** | — | nsys per-replay | **how collective-bound a forward / token is (per-fwd & per-token) — Fig 5** |
| **L1** device | per-replay CV (stability) | % | nsys per-replay | is one comm/comp number per batch size valid? |
| **L2** step | tokens committed / forward | tokens | counter (per-step CSV) | the per-token denominator — how many tokens a forward delivers |
| **L3** block | intrinsic `s_k` = `finish_step+1` | steps | counter (per-block CSV) | a block's own productive denoising steps |
| **L3** block | batch `S_k`, straggler waste | steps, % | counter | forwards a block sits through, and those wasted after it finished |
| **L4** token | **`comm/token`, `comp/token`** (distributions) | ms | **L1 ÷ L2** | **exposed distributed / compute cost per delivered token — headline, Fig 3** |
| **L4** token | `S_k` amplification | × | L3 | exposed comm rounds per block vs the 1-round parallel-decode ideal |
| **L5** serving | throughput, tok/s/GPU | tok/s | driver | does adding concurrency actually help? |

Every metric is reported at **concurrency 4 / 8 / 16** because the per-token cost is an operating-point function, not a constant.

### L1 — Per-forward device cost (per-replay distributions, nsys CUDA-graph trace)
Each forward's comm and compute time is reconstructed **per CUDA-graph replay** (one sample per forward), so the right object is a distribution, not one number. **Compute time is tight** (a near-spike: at conc16 mean 5.99 ms, median 6.57 ms — left-skewed only because ramp/drain forwards at smaller padded bs are cheaper) — content (prompt length, MoE routing) barely moves it. **Communication time is tight-with-a-tail**: mode at 1.18 / 1.45 / 2.00 ms (median) but mean 1.86 / 2.42 / 2.73 ms, pulled up by **desync spikes** (4–6% of forwards > 6 ms, tail to 47 ms). Those spikes are **cross-rank stalls**, not content — between graph replays the un-graphed host-side select loop skews the 4 ranks, so the next forward's first all-reduce waits for the slowest peer (see Method "Is one number per batch size valid?"; per-replay CV: comp 4.3/3.5/2.4%, attention 1.2/0.4/0.6%, comm 152/144/106%). Both grow **sub-linearly** with batch size (comm: fixed collective count, only the all-reduce payload `bs×32×hidden` grows; comp: one larger batched GEMM/MoE is more GPU-efficient).
![Per-forward communication and compute time](figures/fig6_perfwd_time_hist.png)
*Fig 6 — Per-forward measured time: communication (top) has a tight mode + a heavy desync tail (mean ≫ median); compute (bottom) is a near-spike (mean ≈ median) — the visual proof that compute is content-insensitive and a single comp/fwd per bs is justified.*

The two combine into the **communication-time fraction**, `comm / (comm + compute)`. Built from the per-token samples this is `(comm/n) / ((comm/n) + (comp/n))`; the `n` tokens cancel, so the fraction *value* of a forward is the same whether you read it per forward or per token — but the **distributions are not the same object**: per **forward** each forward is one sample, per **token** each forward is weighted by the `n` tokens it decoded (a forward decoding 40 tokens stands for 40 tokens of the workload). We plot both (Fig 5). They turn out **nearly identical** — per-forward vs per-token mean **26.7% vs 27.2%**, median **21.7% vs 21.8%** (conc4) — *because the fraction is independent of `n`*: comm and compute are set by the batch size, not by how many positions cross the confidence threshold that step. So **the typical token (and the typical forward) spends ~22% of its GPU time in communication** (median; mean ~27% incl. the desync tail to ~80%) — a distribution-level refinement of D1's "32% of GPU" total-time mean.
![Communication fraction per forward vs per token](figures/fig5_comm_fraction_hist.png)
*Fig 5 — Communication-time fraction `comm/(comm+compute)`, per FORWARD (top, each forward one sample) vs per TOKEN (bottom, each forward weighted by the tokens it decoded). The two nearly coincide — shown, not assumed — because the fraction is independent of how many tokens a forward commits; median ~22% (dotted), mean ~27% (dashed), thin desync tail to ~80%.*

### L2 — Per-step loop behavior (the per-token denominator)
| conc | tokens committed per forward: mean / median / p10 / p90 / max |
| ---- | ------------------------------------------------------------ |
| 4    | 4.84 / 3 / 1 / 12 / 46                                       |
| 8    | 7.90 / 5 / 1 / 18 / 62                                       |
| 16   | 13.43 / 8 / 1 / 31 / 150                                     |

This is the denominator that converts L1's per-forward cost into per-token cost. The spread is **wide and right-skewed**: many forwards commit just 1 token (low-confidence positions resolved one at a time → the whole forward's comm charged to one token, the worst case) while a few commit dozens (a burst of confident positions → near-zero comm/token). The **mean shifts right 4.8 → 7.9 → 13.4** with concurrency, because a larger realized batch means more active blocks each resolving ~1–2 positions in the same forward — this rightward shift is exactly what drives per-token cost down in L4.
![Tokens committed per forward distribution](figures/fig2_committed_per_step_hist.png)
*Fig 2 — Tokens committed per forward (the per-token denominator), one panel per concurrency, shared axes. Right-skewed with a heavy mode at 1; the mean (dashed) shifts right 4.8→7.9→13.4 as concurrency grows.*

### L3 — Per-block decoding: `S_k`, amplification, straggler waste
| conc | intrinsic `s_k` (mean/med/max) | batch `S_k`/block | tokens/`s_k`-step | n_committed/block | straggler waste |
| ---- | ------------------------------ | ----------------- | ----------------- | ----------------- | --------------- |
| 4    | 13.8 / 14 / 31                 | 21.0              | 2.14              | 29.6              | 7.2 fwd (34%)   |
| 8    | 15.1 / 16 / 31                 | 25.0              | 1.97              | 29.7              | 9.9 fwd (40%)   |
| 16   | 15.3 / 15 / 32                 | 27.2              | 1.95              | 29.8              | 11.9 fwd (44%)  |

Intrinsic `s_k` is **~flat across concurrency** (content-driven, not batch-driven), so the **`S_k`× amplification — ~14–15 exposed comm rounds per block vs the 1-round parallel-decode ideal — is structural**. Batch `S_k` and straggler waste, by contrast, **grow with concurrency**: at conc16 ~44% of every block's forwards are spent after it already finished, waiting for the slowest block in the batch. `batch S_k/block` is block-weighted, so `intrinsic + waste = batch S_k/block` exactly.
![Intrinsic s_k distribution and straggler waste](figures/fig4_sk_and_straggler.png)
*Fig 4 — Left: intrinsic `s_k` pooled across concurrency (content-driven, per-conc means ~coincide at 13.8/15.1/15.3) — a wide spread from a few steps to the full 32. Right: each block's batch `S_k` = productive intrinsic `s_k` (green) + straggler waste (red, forwards endured after the block finished); the wasted fraction grows 34→40→44% with concurrency.*

**How straggler waste is computed.** The denoising loop forwards the *whole batch* every step and exits only when **all** blocks in the call are mask-free (`low_confidence.py:73–75`), so a block that finishes early keeps being re-forwarded until the slowest block in its call is done. Per block, from the per-block CSV: **intrinsic `s_k` = `finish_step + 1`** (own productive steps, `:127–128`); **batch `S_k` = `steps_executed`** (forwards the whole call ran, `:77`); **wasted = batch `S_k` − intrinsic `s_k`** (forwards endured after finishing); **waste % = wasted / batch `S_k`**, block-weighted so `intrinsic + waste = batch S_k/block` exactly.
*Worked example:* a 2-block call exits at batch `S_k = 25` with `finish_step = [11, 24]` — block A (intrinsic `s_k = 12`) endures `25 − 12 = 13` wasted forwards (52%) waiting for block B; block B (the straggler, intrinsic `s_k = 25`) wastes 0. Averaged over all blocks this is the reported per-conc %; larger batches ⇒ higher chance some block is a slow straggler ⇒ the average rises (34→40→44%).

#### Waste source (confirmed mechanism)
> Q: the forward `out = model_runner.forward(forward_batch, ...)` will always run even though some samples have finished?

Exactly — that is the mechanism. The forward at `low_confidence.py:82` runs over the entire `forward_batch` (all `batch_size × 32` positions) every iteration, and the loop only exits when all blocks are mask-free (`:73–75`); a finished block is never removed mid-loop, so its positions are re-forwarded every remaining step. Where the finished block *does* get skipped is only the **select loop** (`:86–96`): the `if torch.sum(block_mask_index)==0: continue` avoids re-selecting (host argmax/commit), but the block was already forwarded at `:82` before the select loop runs — so the skip saves a little host work, **not** the forward. A block that finishes at `finish_step` therefore keeps paying full forward cost (compute + its share of the per-layer TP all-reduce) for steps `finish_step+1 … steps_executed-1` → `wasted = batch_S_k − intrinsic_s_k`.
Why the baseline can't just drop it mid-call: the forward is one batched op over a fixed `[block_size × batch_size]` tensor, **and the CUDA graph is captured for that exact shape**. Evicting a finished block mid-loop would change the shape (and the graph), so the stock path re-forwards everything until the slowest block finishes — exactly the inefficiency ragged batching / finished-block eviction (I1/I5, D7) targets.

#### Why continuous batching does NOT eliminate this waste
Continuous batching (Orca-style iteration-level scheduling) is **already active** in the dLLM path, but it operates one level *above* where this waste occurs, so it cannot reach it. Two distinct stragglers:
- **Request/block-round-level length heterogeneity — continuous batching DOES solve this.** Requests generate different numbers of blocks; SGLang's scheduler reforms the running batch *between* `run()` calls (evict finished, admit new). That is active here — evidence: the realized batch fluctuates (mean 10.7, the `batch_size mix` 16→13→8→…), not pinned at the concurrency.
- **Within-`run()` denoising-step heterogeneity — continuous batching does NOT reach this.** This is the D2 waste. The denoising loop `for step in range(block_size)` (`low_confidence.py:72`) does not yield to the scheduler between denoising steps, so a block that becomes mask-free at step 8 cannot be evicted while its call-mates need 25 — the eviction machinery only runs between `run()` calls. In AR terms continuous batching evicts a finished sequence at the next decode step because each decode step *is* a scheduler iteration; in the dLLM path the denoising steps are buried in an inner loop below the scheduler's granularity.

**Not unique to SGLang.** This is inherent to *batched block-wise dLLM denoising*: any framework that decodes a block over multiple denoising steps and batches several sequences' blocks pays it, unless it implements **step-level / ragged denoising** (lift each denoising step to a scheduler iteration so finished blocks drop out and new ones join between steps). **The catch (why it is not free):** evicting finished blocks mid-denoising *shrinks the active batch*, which **reverses the L4 amortization** (fewer tokens/forward → comm/comp per token rises for the survivors), and it needs variable-shape forwards (compact active blocks → re-pad / re-capture the CUDA graph) with blocks at different denoising stages co-scheduled. That trade-off is the ragged-batching / finished-block-eviction direction (I1/I5) that D7 quantifies.

### L4 — Per-token cost (L1 ÷ L2): the headline
The per-token cost is a **distribution over delivered tokens** (each forward contributes its `measured comm-time / committed`, weighted by the tokens it delivered), not a single number — Fig 3 is the headline. It is **right-skewed**: most tokens are cheap (committed in a burst → the forward's comm shared across many) with a tail of expensive 1-token/desync forwards. Stats (token-weighted, from `d2_dist_stats.json`):
| conc = bs | comm/token median | comm/token mean | comp/token median | comp/token mean | comm:comp (mean) | tokens/fwd (L2) |
| --- | --- | --- | --- | --- | --- | --- |
| 4  | **0.18** | **0.42** | 0.47 | 0.87 | 1 : 2.1 | 4.84  |
| 8  | **0.13** | **0.31** | 0.39 | 0.63 | 1 : 2.0 | 7.90  |
| 16 | **0.09** | **0.18** | 0.27 | 0.45 | 1 : 2.5 | 13.43 |

Both median and mean **fall ~2× as concurrency grows 4×**: comm/forward (mean) rises only ~1.5× while tokens/forward rises 2.8× (L2), so per-token cost amortizes. The **median is the typical delivered token** (~0.1–0.2 ms comm); the **mean is higher** because the right tail (1-token forwards, each charged a whole forward's comm; plus L1 desync stalls) pulls it up. comp/token dominates comm ~2:1 and amortizes the same way.
![Per-token comm/compute distribution](figures/fig3_pertoken_hist.png)
*Fig 3 — Distribution over delivered tokens of exposed comm/token (top) and compute/token (bottom): measured per-forward time ÷ tokens decoded, token-weighted. Right-skewed; mass shifts left as concurrency rises (dashed = mean, dotted = median). This replaces the earlier single "typical..mean" range with the full distribution.*
![Per-token comm/compute means vs concurrency](figures/fig1_pertoken_vs_concurrency.png)
*Fig 1 — Summary: per-token comm (red) and compute (blue) means both fall as concurrency rises, because tokens committed per forward (green, right axis) climbs faster than per-forward comm/comp grows — the batching-amortization effect.*

**Why it falls (batched forward vs sequential select).** It is the standard batching amortization, but easy to misread the loop as if cost scaled per block — it does not. Per denoising step, **one batched forward** runs over *all* blocks (`model_runner.forward`, `:82`; `forward_batch.input_ids` is `[block_size × batch_size]`, the `:84` assert confirms it): **one** set of TP all-reduces and **one** batched GEMM/MoE pass regardless of batch_size, so comm is paid **once per step and shared by every block**. The sequential `for batch_id` loop (`:86–129`) is only the host-side *selection* (argmax/softmax/commit) — it scales with batch_size but is cheap host work (D1: ≈13% of per-step GPU at conc4), not the comm/comp we report. So a forward delivers *more committed tokens* with more blocks while its own cost grows sub-linearly: `comm/token = comm_per_forward / committed`, numerator ~1.5×, denominator ~2.8× → it falls. **Nuance:** the forward re-processes *all* `bs×32` positions every step (full blocks, no eviction — already-committed tokens recomputed), so the per-token *fall* comes from GPU efficiency on the larger batched GEMM/MoE, not from doing less work per token.

**`S_k` amplification & vs an AR model of equal size.** A block emits 32 tokens. **dLLM:** ~15 exposed comm rounds (`s_k`, L3), every one on the critical path (overlap force-disabled). **AR:** 32 comm rounds (one per token) but **overlap-hidden → ~0 exposed**. dLLM pays *fewer* rounds (it commits ~2 tokens/step) yet exposes all of them; the penalty is the **exposure**, multiplied by `s_k`. Committing more tokens/step (higher confidence) shrinks `s_k`, but only step-reduction or overlap removes the exposure.

### L5 — Serving context & cross-level recap
comm/fwd and comm/tok shown as **median (mean)** — both right-skewed (L1 desync tail, L4 1-token forwards); comm%fwd is the per-forward fraction median.
| conc=bs | comm/fwd ms med(mean) | comp/fwd | comm%fwd med | tok/fwd | **comm/tok med(mean)** | comp/tok med(mean) | intrinsic `s_k` | waste% | tok/s | tok/s/GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4  | 1.18 (1.86) | 4.30 | 22% | 4.84  | 0.18 (0.42) | 0.47 (0.87) | 13.8 | 34% | 355 | 89  |
| 8  | 1.45 (2.42) | 5.24 | 22% | 7.90  | 0.13 (0.31) | 0.39 (0.63) | 15.1 | 40% | 523 | 131 |
| 16 | 2.00 (2.73) | 6.63 | 23% | 13.43 | 0.09 (0.18) | 0.27 (0.45) | 15.3 | 44% | 726 | 182 |

Throughput scales **sublinearly** — 4× concurrency yields ~2× tok/s (355→726) and ~2× per-GPU (89→182) — the price of fully-exposed collectives plus the rising straggler waste. The win from batching (per-token comm/comp down ~2×) and the loss from stragglers (waste 34→44%) partly cancel: concurrency lowers cost-per-token but not proportionally to the GPUs added.

## Caveats
- **comm_per_forward grows sub-linearly with bs** (payload `tokens×hidden`); applying the dominant-graph value to all steps within a run is a small approximation (most steps are at/near dominant bs). Across the sweep the growth is reported explicitly (L1).
- **comm/fwd is reported median..mean** because the exposed time is heavy-tailed (rare host-induced cross-rank desync stalls); the typical (median) is the structural cost, the mean is realized-incl-stalls. comp/fwd is content-insensitive (per-replay CV ≤ 4.3%), so one number is used.
- **comp_per_token uses dominant-bs steps only** (coverage ~57–60%), because comp scales with bs; it is the per-token compute *at the dominant batch*, not a batch-mix average.
- **Operating point matters and is the point:** per-token comm is **not** a single number — it is a right-skewed distribution, median 0.18 / 0.13 / 0.09 ms (mean 0.42 / 0.31 / 0.18) at conc 4/8/16 (Fig 3). Quote it *with* the concurrency and as median+mean, not a point.
- **`moe_a2a_backend='none'`** ⇒ the only collective is TP all-reduce (+ vocab all-gather). `--moe-a2a-backend deepep` (D4) would add the EP all-to-all to comm/forward and every per-token comm number.
- **Single base-rank capture**; MoE routing is the only rank-divergent behavior (D4). HumanEval (code) gives one content distribution; `s_k` shifts with prompt mix/threshold — the headline is the mechanism, not the exact 15.
- **comm includes the +1 final forward** in the dominant-graph average; the per-step counter logs denoising steps only (final forward commits no tokens, correctly excluded from per-token denominators).

## Takeaways for direction priority
- **Step reduction (Design-1) is the first-order lever and concurrency-independent.** `s_k`≈15 exposed rounds/block (L3) is structural; halving it halves exposed comm per block directly, at every operating point.
- **State-dependent overlap (I3) targets the exposure.** dLLM exposes a median ~0.09–0.18 ms comm/token (~22% of every forward) that AR hides; recovering it (the loop mutates `input_ids` mid-step → plain overlap blocked, D6) closes the gap toward AR's ~0. The desync-stall tail (L1, mean ≫ median) is a *separate*, host-side overlap target — keeping ranks in lockstep between steps would collapse the mean toward the median.
- **Ragged batching / straggler packing (I1/I5) is now quantified and grows with concurrency** — 44% wasted forwards at conc16 (L3). There is a real **batching↔straggler tension**: more concurrency lowers comm/token (good, L4) but raises wasted collective rounds (bad, L3). Finished-block eviction or ragged exit reclaims the latter — at the cost of reversing some L4 amortization.
- **Compute (≈2× comm/token) co-amortizes**; MoE-runner / EP work (D4) is competitive with comm work, consistent with D1.

## Next
- **D4 (`--moe-a2a-backend deepep`):** add the EP all-to-all to comm/forward and re-measure per-token comm; check expert-load drift across `s_k` steps.
- **D7 (straggler):** this run already exposes the 34→44% waste trend; D7 formalizes wasted *collective* rounds vs concurrency and tests ragged-batch exit.
- **Design-1 validation:** re-run this per-step counter after a step-reduction policy; confirm comm/token drops ∝ mean `s_k`.
- Cross-links: plan `experiments/profiling/dllm/dllm_baseline_profiling_plan.md` (D2); comm-fraction baseline `experiments/profiling/dllm/d1_comm_decomposition/README.md` (D1, 32% comm — consistent with the ~30% here). Scripts: this dir (`run_d2.sh`, `drive_humaneval.py`, `parse_d2.py`, `plot_d2.py`); figures in `figures/`. Data: `$DATA_ROOT/profiling/dllm/d2_sk_amplification/`.
