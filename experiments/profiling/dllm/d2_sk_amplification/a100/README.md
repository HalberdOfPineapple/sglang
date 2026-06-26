# Experiment D2 (A100 extension) — Projected-NVLink Comm Fraction per Output Token, from Communication Volume (LLaDA2.0-mini, 4×A100 80GB PCIe; comm projected to A100 NVLink)

Extends the H100 D2 experiment ([`../h100/README.md`](../h100/README.md)) to a machine without NVLink — **4×A100 80GB PCIe** — and adds a **batch-size=1** point. Because the raw all-reduce time over PCIe/`SYS` is interconnect-bound and **meaningless to quote as the dLLM's comm cost** (of course NVLink is faster), we do **not** report measured PCIe comm. Instead we keep the two quantities that *are* meaningful from this box — the **analytic communication volume** (shape-deterministic) and the **real measured A100 compute time** (interconnect-independent) — and **project the communication time from volume onto A100 NVLink** (`projected_comm = bus_traffic / busbw`). The headline is then the **same metric D2 reports — the communication-time fraction `comm/(comm+compute)` per output token — but with comm *projected* and compute *measured on A100***: i.e. the comm/compute balance this A100 box *would* show if it had NVLink. Swept over **bs 1/4/8/16** on HumanEval; the §3 per-step counter ties every forward to the tokens it decoded.

> **Read against the H100 sibling, do not over-read the absolute %.** The projection divides volume by bandwidth, so it is a **bandwidth model**. The 41 TP all-reduces per forward are *small* messages (`bs×128 KB`) and partly **latency-bound**, so `volume/busbw` is a **lower bound** on comm — and hence the projected comm fraction below (median **1.4 / 3.1 / 5.0 / 7.6%** at bs 1/4/8/16) is a **floor**. The H100 sibling, which *measures* the same collectives on real NVLink, sees a **latency-bound 22–32%** (its headline). Using those measured collective times as the A100-NVLink anchor gives a **latency-aware ~13–14%** at bs≥4 (Caveats). So the true A100-on-NVLink comm fraction is **bracketed ≈ [7.7%, 14%] at bs16** — the requested volume/bandwidth number is the low end.

## Summary

Projected comm (volume / **240 GB/s** A100-NVLink busbw) over the stock `LowConfidence` path, TP4/EP4, CUDA graph ON, overlap force-disabled; compute **measured** per CUDA-graph replay on A100; both joined to the §3 per-step counter on HumanEval at sustained concurrency = bs. Findings (operating point = bs 1/4/8/16):
1. **Projected comm is a small, batch-growing fraction of the A100 forward — median 1.4 / 3.1 / 5.0 / 7.6% per forward** (mean 1.4 / 2.9 / 4.6 / 6.9%; Fig 5), i.e. comm:compute ≈ **1:70 → 1:13** per delivered token as bs grows. The fraction *rises* with batch because comm **volume scales linearly with bs** (`bus_traffic = 250 MB/fwd` at bs16) while measured compute amortizes per token — the opposite direction from a fixed-cost view. **This is a bandwidth-floor; the real (latency-bound) number measured on H100 is 22–32%** — see the blockquote and Caveats.
2. **Per-token compute amortizes; projected per-token comm barely moves (Fig 1, 3).** Token-weighted compute/token **2.27 → 1.61 → 1.19 → 0.84 ms** (falls ~2.7× as tokens/forward climb 2.0 → 4.6 → 7.9 → 13.7), while projected comm/token is **0.033 → 0.049 → 0.059 → 0.064 ms** (nearly flat — comm volume grows ∝ bs, cancelling the per-token amortization). So batching cuts per-token *compute* but not per-token *comm*.
3. **The communication is two collectives of comparable volume.** Per forward: **41 TP all-reduces** of `[bs×32, 2048]` bf16 (`bus = 129 MB` at bs16) **+ 1 vocab all-gather** of `[bs×32, 157184]` (`bus = 121 MB` at bs16) — total **250 MB/forward bus traffic** at bs16, split ~52/48 all-reduce/all-gather. The vocab all-gather, negligible on H100 (1.4% of GPU there), is **half the comm volume** and matters once projected at finite bandwidth.
4. **`S_k` and straggler structure reproduce D2 exactly (content-driven, interconnect-independent).** Intrinsic `s_k` **mean 14.7 / 13.7 / 14.9 / 15.0**, ~flat — a block pays **~15 exposed comm rounds** to emit 32 tokens vs the 1-round parallel-decode ideal (**~15× round amplification**); in-batch **straggler waste rises 0 → 36 → 39 → 46%** of each block's forwards as bs grows (0% at bs1: a single block never waits).
5. **On the real PCIe hardware, batching gives *zero* throughput scaling — 74 / 75 / 70 / 69 tok/s, flat** (vs the H100 sibling's 355 → 726). On PCIe the forward is comm-bound (raw measured comm is 72–94% of the forward), and comm grows ∝ bs, so larger batches buy no tok/s. **This is the actual-hardware penalty; the projected-NVLink scenario (comm ≤ 8% of compute) would instead be compute-bound and scale like the H100 run** — the whole point of projecting.

Net: projected onto NVLink, the dLLM's **TP all-reduce comm is a modest fraction of compute on a *bandwidth* basis** (≤ 8% floor, ~13–14% latency-aware), so **interconnect bandwidth is not the first-order lever** — the levers are the **collective *count* × `S_k`** (Design-1 step reduction) and the **exposure** (I3 overlap), exactly as D2 concluded, plus the **vocab all-gather volume** (E-b) which is now half the comm bytes. The straggler waste (→46%) is the I1/I5 lever, unchanged by interconnect.

## Setup

### Hardware & software
- **GPUs:** 4× NVIDIA **A100 80GB PCIe**, **no NVLink**. `nvidia-smi topo -m`: GPU0↔{1,2,3} = `SYS` (cross-socket, traverses the CPU SMP/UPI link — the slowest hop), GPU1↔2↔3 = `NODE`/`PHB` (same NUMA, via PCIe host bridge). So the TP4 all-reduce ring crosses the `SYS` hop. NUMA: GPU0 on node0, GPU1–3 on node1. **This is why raw comm time is not reported — it is dominated by the `SYS` hop, an artifact of this box, not of the dLLM.**
- **Software:** conda env `sglang` (Python 3.10), SGLang git `194400263`, **nsys 2025.1.1** (bundled with the `eval_310` env's nsight-compute). This nsys build has **no `nccl` trace plugin**; we trace `-t cuda,nvtx` and classify NCCL by **device-kernel name** (`ncclDevKernel_AllReduce_…RING_LL`, `ncclDevKernel_AllGather_RING_LL`) from `CUPTI_ACTIVITY_KIND_KERNEL` — comm volume is analytic so no NCCL byte trace is needed. The all-reduce uses **NCCL RING_LL** (custom one-shot all-reduce falls back to NCCL over PCIe). nsys 2025.1.1 also lacks the `graphId` kernel column, so graph replays are grouped on `graphNodeId` (parser `_graph_nodes`/`_bucket_graphs`).
- **A100 NVLink reference (for the projection):** A100 SXM **NVLink3** = 12 links × 25 GB/s/dir = **300 GB/s/dir** (600 GB/s/GPU bidirectional). Achievable 4-GPU all-reduce busbw (nccl-tests class) ≈ **240 GB/s** (headline); per-link unidirectional ceiling **300 GB/s** (sensitivity). Chosen so the projected comm divides by *A100* NVLink bandwidth and is formed against the *A100*-measured compute (per the experiment's intent).
- **Model:** `inclusionAI/LLaDA2.0-mini` — `llada2_moe`, 20 layers, hidden 2048, **256 experts / top-8**, vocab 157184, bf16. `block_size=32`, `mask_id=156895`.

### Parallelism / runtime config (confirmed from server log)
| Setting | Value |
| --- | --- |
| dLLM algorithm | `LowConfidence`, threshold 0.95 |
| TP / EP | 4 / 4 |
| MoE A2A backend | `none` (TP all-reduce; no EP all-to-all) |
| Attention backend | `flashinfer` |
| `mem_fraction_static` | 0.7 |
| `max_running_requests` | = bs (1 / 4 / 8 / 16) per run |
| `page_size` | 32 (= block_size) |
| overlap schedule | disabled (`disable_overlap_schedule=True`) → comm fully exposed |
| full CUDA graph | ON; `cuda_graph_bs=[1,2,4,8,12,16,…]` |
| dtype / KV dtype | bf16 / bf16 |

### Workload
**HumanEval** (`/cephfs/shared/wxli/human-eval`), first **20 problems** cycled by an asyncio driver (`max_new_tokens=256`, `temperature=0`), sustaining a full running batch = bs. Real code-completion prompts give a content-driven `s_k` spread. **bs=1 is host-bound on the real hardware** (single block, no batch), but its **volume and compute are well-defined**, which is exactly why the projection (not the PCIe measurement) is the right lens there.

### Runs
| Tag | bs / max-running | reqs | forwards / blocks | tok/s (real PCIe) |
| --- | --- | --- | --- | --- |
| `d2_a100_tp4_c1` | 1 | 8 | 1041 / 71 | 74 |
| `d2_a100_tp4_c4` | 4 | 16 | 993 / 154 | 75 |
| `d2_a100_tp4_c8` | 8 | 32 | 1208 / 320 | 70 |
| `d2_a100_tp4_c16` | 16 | 64 | 1362 / 628 | 69 |

Artifacts: `$DATA_ROOT/profiling/dllm/d2_sk_amplification/a100/` (`DATA_ROOT` default `/cephfs/shared/wxli/sglang-dllm`): `profiles/<tag>.nsys-rep`+`.sqlite`+`_cuda_gpu_kern_sum.csv`; `logs/<tag>_blocks.csv`, `<tag>_blocks_perstep.csv`, `<tag>_summary.txt`, `<tag>_blocks_a100metrics.json`, `<tag>_drive.log`, `d2_a100_sweep_summary.txt`, `a100_dist_stats.json`.

## Method & tooling

### What is measured vs projected vs analytic
- **Measured (A100, real):** the **compute time per forward** — reconstructed per CUDA-graph replay from the nsys `.sqlite` (every `graphNodeId` fires once per replay; sort a node's instances by start time → index by replay; replay *i*'s compute = Σ of every non-collective kernel's *i*-th instance). Content-insensitive (per-replay CV ≤ 2.4%, attention CV ≤ 2.7%), so the median is a clean per-bs number. **The raw PCIe comm time is also reconstructed but discarded** (interconnect-bound; reported once per run as a sanity line only).
- **Analytic (deterministic):** the **communication volume**. Per forward: **all-reduce** = `A_ar × bs×block×hidden×dtype` with `A_ar = 2×layers+1 = 41` (verified against the trace: 41.0 all-reduce instances/replay) and ring **bus traffic** `= 2(N-1)/N × msg` (N=4 → ×1.5); **vocab all-gather** = `bs×block×vocab×dtype` gathered, ring bus traffic `= (N-1)/N × gathered` (×0.75). bf16 = 2 B.
- **Projected (A100 NVLink):** `projected_comm = bus_traffic / busbw`, `busbw = 240 GB/s` (achievable; 300 GB/s sensitivity). The **comm fraction** = `projected_comm / (projected_comm + measured_compute)`; **per-token** = each quantity ÷ the tokens that forward committed (§3 counter), token-weighted.

### Per-step counter (the per-token denominator) — unchanged from D2
The §3 counter (`DllmStepCounter`, `SGLANG_DLLM_PROFILE=1`, rank-0) writes a per-step CSV `(call_id, step, batch_size, n_active, committed)` (one row per denoising forward) and a per-block CSV `(…, S_k, finish_step, n_committed)`. Baseline path bit-identical when off. `committed` is the per-token denominator; `finish_step+1` = intrinsic `s_k`; `steps_executed` = batch `S_k`; `batch S_k − intrinsic s_k` = straggler waste.

### Exact reproduction
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
# full sweep (launches the server under nsys once per batch size):
bash experiments/profiling/dllm/d2_sk_amplification/a100/run_a100.sh        # CONC_LIST="1 4 8 16"
# one point:   CONC_LIST=8 bash .../run_a100.sh
# re-derive a saved run (no GPU):
python experiments/profiling/dllm/d2_sk_amplification/a100/parse_a100.py \
  $DATA_ROOT/.../a100/logs/d2_a100_tp4_c8_blocks $DATA_ROOT/.../a100/profiles/d2_a100_tp4_c8
python experiments/profiling/dllm/d2_sk_amplification/a100/parse_a100.py --sweep $DATA_ROOT/.../a100/logs
# regenerate the 6 figures + a100_dist_stats.json (no GPU; reads saved CSVs + .sqlite):
python experiments/profiling/dllm/d2_sk_amplification/a100/plot_a100.py
```
The NVLink bandwidth is a constant in `parse_a100.py` (`BUSBW_REFS`/`BUSBW_HEADLINE`); change it to re-project.

## Results

All per-forward measured-compute / projected-comm joined to the per-step counter (rep rank dev=0). Per-token and per-forward quantities are **distributions** (Fig 3, 5, 6); the comm-fraction distribution is tight because projected comm is deterministic per bs and measured compute is a near-spike.

### Metrics measured (skeleton — read this first)
| level | metric | unit | source | answers |
| --- | --- | --- | --- | --- |
| **L1** device | compute / forward | ms | nsys per-replay (**measured A100**) | real compute cost of one denoising forward |
| **L1** device | comm volume / forward | MB | analytic | shape-deterministic bytes moved by the collectives |
| **L1** device | projected comm / forward | ms | volume ÷ 240 GB/s | comm time this box *would* pay on NVLink |
| **L1** device | **comm fraction** = comm/(comm+comp) | — | projected comm + measured comp | **projected collective-boundness of a forward (Fig 5)** |
| **L2** step | tokens committed / forward | tokens | counter | the per-token denominator |
| **L3** block | intrinsic `s_k`, batch `S_k`, waste | steps, % | counter | exposed rounds/block + straggler waste (Fig 4) |
| **L4** token | **comm/token, comp/token** | ms | L1 ÷ L2 | projected comm & measured compute per delivered token (Fig 1, 3) |
| **L5** serving | throughput (real PCIe) | tok/s | driver | actual-hardware scaling (interconnect-bound) |

### L1 / L4 — per-forward and per-token (projected comm, measured compute)
| bs | comm vol (MB bus) | proj comm/fwd (ms) | meas comp/fwd (ms) | **comm frac** med (mean) | comm/tok (ms) | comp/tok (ms) | comm:comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15.6 | 0.065 | 4.551 | **1.4%** (1.4%) | 0.033 | 2.275 | 1 : 70 |
| 4 | 62.4 | 0.260 | 8.082 | **3.1%** (2.9%) | 0.049 | 1.614 | 1 : 33 |
| 8 | 124.8 | 0.520 | 9.776 | **5.0%** (4.6%) | 0.059 | 1.186 | 1 : 20 |
| 16 | 249.7 | 1.040 | 12.472 | **7.6%** (6.9%) | 0.064 | 0.835 | 1 : 13 |

Comm fraction is **per-forward median (mean)** from `a100_dist_stats.json`; per-token fraction is within 0.1% of per-forward (the fraction is ⊥ tokens committed — comm and compute are both set by bs, not by how many positions cross threshold — so per-forward and per-token coincide, shown in Fig 5). Compute/forward **grows sub-linearly** with bs (one larger batched GEMM/MoE is more efficient) and compute/token **falls** as tokens/forward climb — the standard batching amortization. Projected comm/forward **grows linearly** with bs (pure volume) so comm/token is ~flat; hence the comm fraction *rises* with batch.
![Per-token compute and projected comm vs batch size](figures/fig1_pertoken_vs_bs.png)
*Fig 1 — Token-weighted compute/token (measured A100, blue) falls 2.27→0.84 ms as tokens/forward (green) climb 2.0→13.7; projected comm/token (red) stays ~0.03–0.06 ms. Batching amortizes compute, not comm.*
![Per-token comm and compute distributions](figures/fig3_pertoken_hist.png)
*Fig 3 — Distribution over delivered tokens of projected comm/token (top, ms) and measured compute/token (bottom, ms), token-weighted. Compute mass shifts left with bs (amortization); projected comm mass barely moves.*
![Projected communication-time fraction](figures/fig5_comm_fraction_hist.png)
*Fig 5 — Projected comm fraction `comm/(comm+compute)` per FORWARD (top) vs per TOKEN (bottom). Tight (projected comm deterministic per bs, measured compute a near-spike); mean rises 1.4→6.9% with batch. This is the **bandwidth-floor**; the H100 sibling measures 22–32% (latency-bound).*

### L1 — per-forward time (projected comm vs measured compute)
![Per-forward projected comm and measured compute](figures/fig6_perfwd_time_hist.png)
*Fig 6 — Per-forward projected comm (top; discrete because it is volume/busbw at each bs in the running mix) and measured A100 compute (bottom; a content-insensitive near-spike, the visual proof one comp/fwd per bs is justified). Compute (4.5→12.5 ms) dwarfs projected comm (0.07→1.04 ms).*

### L2 — tokens committed per forward (the denominator)
| bs | mean | median | p10 / p90 / max |
| --- | --- | --- | --- |
| 1 | 2.00 | 1 | 1 / 4 / 23 |
| 4 | 4.58 | 3 | 1 / — / — |
| 8 | 7.85 | 5 | 1 / — / — |
| 16 | 13.73 | 8 | 1 / 32 / 119 |

Right-skewed with a heavy mode at 1 (low-confidence positions resolved one at a time); the mean shifts right with bs (more active blocks each resolve ~1–2 positions per forward), which is what amortizes compute/token in L4.
![Tokens committed per forward](figures/fig2_committed_per_step_hist.png)
*Fig 2 — Tokens committed per forward, one panel per bs (shared axes). Heavy mode at 1; mean shifts right 2.0→13.7 with bs.*

### L3 — per-block decoding: `s_k`, amplification, straggler waste
| bs | intrinsic `s_k` (mean) | batch `S_k`/block (mean) | straggler waste |
| --- | --- | --- | --- |
| 1 | 14.7 | 14.7 | 0% |
| 4 | 13.7 | 21.2 | 36% |
| 8 | 14.9 | 24.6 | 39% |
| 16 | 15.0 | 27.6 | 46% |

Intrinsic `s_k` is **content-driven and ~flat** across bs (and matches the H100 sibling's 13.8/15.1/15.3) — so the **~15× round amplification** (15 exposed comm rounds/block vs the 1-round parallel-decode ideal) is structural and interconnect-independent. Straggler waste (the batch runs until its slowest block is mask-free) **rises 0→46%** with bs — 0% at bs1 (a lone block never waits), then climbing as more blocks mean a likelier slow straggler. `batch S_k/block` is block-weighted so `intrinsic + waste = batch S_k` exactly.
![Intrinsic s_k and straggler waste](figures/fig4_sk_and_straggler.png)
*Fig 4 — Left: intrinsic `s_k` pooled across bs (content-driven; per-bs means ~coincide at ~15). Right: each block's batch `S_k` = productive intrinsic `s_k` (green) + straggler waste (red); wasted fraction grows 0→46%.*

### L5 — serving context (real A100/PCIe hardware)
Throughput is **flat at 74 / 75 / 70 / 69 tok/s** across bs 1/4/8/16 (18 / 75 / 70 / 69 tok/s/GPU-ish; ~17–19 tok/s/GPU) — **batching buys no throughput** on PCIe, vs the H100 sibling's 355→726 tok/s (sublinear but real scaling). On the real hardware the forward is comm-bound (the discarded PCIe measurement: comm is 72→94% of the forward as bs grows, the all-reduce alone is 140→2073 µs/call over `SYS`), and comm grows ∝ bs, so a 16× batch yields ~0× tok/s. **Projected to NVLink the forward is compute-bound** (comm ≤ 8% of compute, L1), so the projected-NVLink serving point would scale with compute like the H100 run — the reason to project rather than to quote the PCIe number.

## Caveats
- **The headline comm fraction is a *bandwidth floor* — flag this loudly.** `projected_comm = volume / busbw` assumes the collectives saturate bandwidth, but the **41 TP all-reduces are small** (`bs×128 KB`) and partly **latency-bound** (per-call latency ≫ `msg/busbw` at these sizes). So the projected fraction (median **1.4 / 3.1 / 5.0 / 7.6%**) **under-states** comm. The **H100 sibling measures the same collectives on real NVLink and sees 22–32%** (latency-bound). A **latency-aware** estimate using the H100-measured collective times as the A100-NVLink anchor (NVLink3 latency ≈ NVLink4 to first order; all-reduce ~25–39 µs/call) gives total comm/forward ≈ 1.18 / 1.45 / 2.00 ms at bs 4/8/16, i.e. comm fraction **≈ 12.7 / 12.9 / 13.8%** against the A100 compute — so the **true A100-on-NVLink comm fraction is bracketed ≈ [volume-floor 7.7%, latency-aware 14%] at bs16**. The requested volume/bandwidth figure is the conservative low end; the lever it points to (collective *count*/latency, not bandwidth) is the same either way.
- **A100 NVLink bandwidth is an assumption (240 GB/s achievable).** At the 300 GB/s peak the projected comm fraction drops ~20% (bs16: 7.7%→6.3%); the bracket and the takeaways are unchanged. The number is parameterized (`BUSBW_REFS`).
- **Vocab all-gather volume model** assumes the LM head gathers logits for all `bs×block` positions over the full vocab; it cross-checks against the H100-measured all-gather (e.g. bs16 model 120.7 MB bus vs H100 measured 0.410 ms ≈ 294 GB/s effective — consistent with a bandwidth-bound big message). It is **half the comm volume**, unlike on H100 where it was 1.4% of GPU time.
- **Compute is A100, not H100** — A100 compute/forward (4.6→12.5 ms) is ~1.5–2× the H100 sibling's (4.4→6.7 ms), so the *denominator* of the comm fraction is larger here; that is intentional (we want the balance *on this A100 box*), but it is a second reason the projected fraction sits below H100's.
- **bs=1 is host-bound on the real hardware** (single block, throughput 74 tok/s); its volume and compute are well-defined, but its per-forward compute (4.55 ms) carries relatively more host-visible overhead than the graphed steady state.
- **`moe_a2a_backend='none'`** ⇒ the only collectives are TP all-reduce + vocab all-gather (no EP all-to-all). **Single base-rank capture**; MoE routing is the only rank-divergent behavior. nsys 2025.1.1 has no `nccl` byte trace, but volume is analytic so this does not affect the result.

## Takeaways for direction priority
- **Interconnect *bandwidth* is not the first-order dLLM lever — even projected onto NVLink, comm is ≤ 8% (floor) / ~14% (latency-aware) of compute.** The first-order levers are the **collective count × `s_k`** (Design-1 step reduction: ~15 exposed rounds/block × 41 all-reduces) and the **exposure** (I3 state-dependent overlap), exactly as the H100 D2 concluded — the A100 projection does not change the ranking, it confirms it from the bandwidth side.
- **The vocab all-gather is now a first-class comm target (E-b).** It is **half the projected comm volume** (it was 1.4% of GPU on NVLink); at lower bandwidth or larger vocab it dominates, so vocab-parallel-gather reduction / fusion is worth more off-NVLink than the H100 picture suggested.
- **Straggler waste (→46%) and `s_k`≈15 are interconnect-independent** (I1/I5 ragged batching, Design-1) — the A100 run reproduces the H100 trend, so these levers are hardware-portable.
- **PCIe is a hard floor for this workload as-is** (flat 70 tok/s, comm-bound) — the engineering takeaway is that **dLLM serving needs NVLink-class interconnect *or* the comm-count/overlap reductions above** to get any batch scaling; bandwidth alone (PCIe→NVLink) flips the bottleneck from comm to compute, after which the D2 levers apply.

## Next
- **Latency-aware projection:** replace `volume/busbw` with `α + msg/busbw` per collective, calibrating `α` from the H100 sibling's per-call latency, to tighten the [7.7%, 14%] bracket into a single A100-NVLink estimate.
- **D4 (`--moe-a2a-backend deepep`)** would add the EP all-to-all to the comm volume and re-project; check whether all-to-all over PCIe/NVLink changes the fraction ranking.
- Cross-links: **H100 sibling** [`../h100/README.md`](../h100/README.md) (measured 22–32% comm fraction — the latency-bound counterpart to this bandwidth floor); plan `experiments/profiling/dllm/dllm_baseline_profiling_plan.md` (D2); comm decomposition `experiments/profiling/dllm/d1_comm_decomposition/README.md` (D1). Scripts: this dir (`run_a100.sh`, `drive_humaneval.py`, `parse_a100.py`, `plot_a100.py`); figures in `figures/`. Data: `$DATA_ROOT/profiling/dllm/d2_sk_amplification/a100/`.
