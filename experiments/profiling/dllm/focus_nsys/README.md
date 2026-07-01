# Experiment F6 — nsys kernel-level bottleneck analysis: why the FOCUS speedup is minor

**Question.** Kernel-FOCUS beats LowConfidence by only **~1.03–1.05×** (F5), while the FOCUS paper reports up to **2.32×** (vs an LMDeploy baseline). Where does the wall-clock go, and why doesn't the real token eviction (redundancy ~0.75, i.e. ~25% fewer tokens through L1-attn..L) translate into a bigger win?

**Setup.** LLaDA2.0-mini, 1×A100 80GB, TP=1, **eager** (`--disable-cuda-graph`), HumanEval-style prompts, sustained batch of 8, `max_new_tokens=128`. nsys 2025.1.1 (the box's validated build — the sglang-env 2025.2.1 has a broken qdstrm importer), `-t cuda,nvtx`, capture bracketed by SGLang's `/start_profile` CUDA_PROFILER; FOCUS/LowConfidence phases tagged via `SGLANG_DLLM_NVTX=1`. Reports: `cuda_gpu_kern_sum`, `nvtx_gpu_proj_sum`, `cuda_gpu_trace`. **Absolute nsys wall is not a throughput measurement** (single batch + profiler overhead) — the *decomposition* (idle %, category mix, per-phase shares, kernel counts) is the signal; throughput numbers come from F5.

## Three findings that together explain the small speedup

### 1. The regime is >50% GPU-idle — launch-bound, not compute-bound
| | GPU busy | span | **% idle (exposed host/launch-gap)** | kernels | avg kernel |
|---|---|---|---|---|---|
| kernel-FOCUS | 2645 ms | 6021 ms | **56.1%** | 131,534 | 20.1 µs |
| LowConfidence | 2821 ms | 5946 ms | **52.5%** | 121,532 | 23.2 µs |

More than half the wall is GPU-**idle** — the eager per-layer/per-expert launch gaps on a tiny model. Any FLOPs reduction (all FOCUS does) can only attack the ~44–48% that is actually GPU-busy, so Amdahl caps the achievable win hard before eviction even enters.

### 2. FOCUS cuts GPU compute, but only ~6% — the MoE kernel doesn't scale with tokens
GPU-busy: FOCUS 2645 ms vs LC 2821 ms = **−6.2%**. Far below the ~25% token eviction, because of *what* dominates:

| category | FOCUS ms | LC ms | Δ | share (LC) |
|---|---|---|---|---|
| **moe/gemm** | 2302.9 | 2472.2 | **−6.8%** | 87.8% |
| norm/elementwise | 189.8 | 182.8 | +3.8% | 6.5% |
| **attention** | 69.8 | 100.5 | **−30.5%** | 3.6% |
| moe-route/topk | 18.6 | 23.9 | −22% | 0.8% |

- The single biggest kernel is **`fused_moe_kernel` (~69% of GPU-busy: 1814 ms FOCUS / 1963 ms LC)** — it drops only **7.6%** for a 25% token cut. The MoE grouped-GEMM over **256 experts / top-8** with few tokens per expert is **occupancy/launch-bound, not per-token-FLOP-bound**: fewer tokens ⇒ same fixed per-expert overhead, so time barely shrinks.
- **Attention is the one piece that scales with tokens** — it drops **30.5%** (≈ the redundancy) — but it's only 2.6–3.6% of GPU time, so it can't move the wall.

### 3. FOCUS gives part of the saving back as structure LowConfidence never pays
Per-NVTX-phase GPU-projected time:

| kernel-FOCUS phase | share | | LowConfidence | share |
|---|---|---|---|---|
| S: `focus_suffix` (L2..L on \|S\|) | **82.8%** | | `dllm_forward.step` | 95.4% |
| **final full forward (KV repop)** | **6.1%** | | `dllm_select.step` | 4.6% |
| A1: `focus_l1_attn` (L1 attn on \|S\|) | 4.8% | | | |
| P: `focus_prefix` (L0+L1 QKV, **full B**) | 4.2% | | | |
| commit | 2.2% | | | |

FOCUS's reduced forward (`focus_suffix`, L2..L on \|S\|) is where the win lives, but it also pays **~17% of its GPU work on structure the stock path skips**: a **mandatory full-block prefix (L0+L1) every step**, a **full-block final forward every block** to repopulate KV (6.1% alone), the A1 phase, and commit. That structural overhead — plus **8% more kernel launches** (the 3-phase split) feeding the launch-bound idle in finding #1 — cancels much of the L2..L eviction saving.

**Net:** −6% GPU compute, partly returned as +structure/+launches in a >50%-idle regime ⇒ the ~1.03–1.05× measured in F5. The nsys spans (6021 vs 5946 ms) corroborate near-parity on this single batch.

## Why it's far from the paper's 2.32× (two regime gaps, not bugs)
1. **~3× less eviction headroom here.** Redundancy on LLaDA2.0-mini + HumanEval is ~0.75 kept (evict ~25%); the paper's redundancy curves evict up to ~90%. FOCUS can only save what it evicts — the FLOPs headroom is intrinsically smaller on this model/workload. (Their baseline is LMDeploy on different, denser models/serving.)
2. **Launch-bound eager small-MoE ≠ compute-bound.** The paper's speedup assumes the MoE GEMMs scale with token count (compute-bound). Here >50% of wall is GPU-idle and the MoE kernel is underutilized, so even the FLOPs FOCUS *does* cut don't convert to wall-clock.

## Where the bottleneck is / what would move it
The bottleneck is the **MoE forward (`fused_moe_kernel`, ~69% of GPU-busy) in an eager, launch-bound, GPU-underutilized regime.** To turn the real FLOPs cut into wall-clock, in priority order:
1. **§C — Phase-S CUDA graph.** Collapse the ~18 eager L2..L layer launches (the 82.8% `focus_suffix`) into one graph replay, directly attacking the 56% idle. The nsys idle fraction is the strongest evidence yet that this is *the* lever. (§B removed the host loops; §C removes the launch gaps.)
2. **A compute-bound operating point** — larger model / longer context / more tokens-per-expert — where the MoE GEMM scales with tokens, so 25% eviction → ~25% MoE-time. That is where the paper's numbers live.
3. **Cut FOCUS's structural overhead** — repopulate KV incrementally instead of a full-block `final forward` (−6.1%), and/or fuse the full-block prefix — reclaiming the ~17% GPU work spent outside Phase S.

## Reproduce
```bash
cd /root/sglang_a100/sglang/experiments/profiling/dllm/focus_nsys
NSYS=/root/miniconda3/envs/eval_310/nsight-compute-2025.1.1/host/target-linux-x64/nsys \
  ./run_nsys.sh both                 # captures kernel-FOCUS + LowConfidence at conc 8
python parse_nsys.py <PROFILES_DIR>   # the tables above
```
Data (mirror): `/cephfs/shared/wxli/sglang-dllm/profiling/dllm/focus_nsys/{profiles,logs}/` — `*.nsys-rep`, `*_{cuda_gpu_kern_sum,nvtx_gpu_proj_sum,cuda_gpu_trace}.csv`, `focus_nsys_summary.txt`.

## Caveats
- **Use nsys 2025.1.1** (`envs/eval_310/...`); the sglang-env 2025.2.1 produced only `.qdstrm` (importer "Unable to retrieve importer version"). No `nccl` trace plugin on this box (irrelevant at TP=1).
- Category buckets are name-heuristic (`parse_nsys.py:_category`); the coarse split (MoE 87% / attn 3% / idle 55%) is robust to edge misclassification.
- `nvtx_gpu_proj_sum` Total Proj Time projects GPU ops onto each range's wall (nested/overlap inflates absolutes); read the **shares**, cross-checked against `cuda_gpu_kern_sum` (MoE 87%) and the F5 phase timing (`s_fwd` 67.5%). All three agree: L2..L MoE dominates.
