# Experiment D1 — Per-step Comm Decomposition (baseline LowConfidence, 4×H100 NVLink)

Re-run of D1 on **NVLink** hardware. The first pass (`notes/experiment_20260619_d1_comm_decomposition.md`) ran on **4×A100 PCIe with no NVLink**, where the TP ring crossed a cross-socket `SYS` hop. Headline question here: **does full NVLink change the comm story, and is the A100 "87% comm" figure real?** Short answer: the production exposed-comm fraction is **32.3%** (not 87%), the 87% was an eager-mode artifact that reproduces on NVLink, and the bottleneck shifts from host-side selection (an A100 bs=1 effect) to TP all-reduce + MoE compute.

## Summary

Faithful production measurement of the stock dLLM path on **NVLink**, at a **GPU-bound** operating point (concurrency=4), with in-graph NCCL exposed via `nsys --cuda-graph-trace=node` (no eager distortion). Four findings:
1. **TP all-reduce is the single exposed collective, large but not dominant.** `AllReduce(TP)` = 30.9% of GPU kernel time (2204 ms / 7137 ms); `AllGather` (vocab LM-head) only 1.4%; **no EP all-to-all** (`moe_a2a_backend=none`). Per forward: 41 all-reduces × ~40 µs = ~1.66 ms (20 layers × {attn-out, MoE-out} + 1), i.e. ~1.74 ms comm/forward. Measured **structurally** from the forward CUDA graph (the only correct per-step method — see Results), **communication is 32.1% of the GPU kernel work in one forward step** at the dominant batch size, rising to **43.2% at the smallest captured bs** (TP all-reduce latency is bs-insensitive while MoE compute shrinks). This matches the 32.3% global GPU-comm because the forward dominates GPU work. Since overlap is force-disabled for dLLM, this comm is **fully exposed**.
2. **The "87% comm" headline is an eager-mode artifact, not interconnect cost.** The eager (`--disable-cuda-graph`) run hits **89.6% comm** on NVLink — reproducing the A100 figure — with `AllReduce` at ~751 µs/call ≈ the A100/`SYS` ~700 µs/call despite ~10× the link bandwidth. That equality proves the inflation is **cross-rank launch-desync spin-wait** (eager launches collectives separately → ranks desync → every all-reduce spins for the slowest peer), which the **CUDA graph removes by lockstep replay** (graph-ON all-reduce = ~40 µs/call). NVLink does make the vocab `AllGather` cheap (~77 µs vs 5.6 ms over A100 `SYS`), but the comm story is set by graph-vs-eager, not by the link.
3. **The host-bound bs=1 bottleneck does NOT survive batching.** At concurrency=4 `dllm_select` is only 1.11 ms GPU + 3.45 ms CPU per step, below the 7.28 ms GPU forward — the step is **forward/compute-bound** (forward 87% / select 13% of per-step GPU). The A100 finding ("~17 ms wall, 92% GPU-idle, host-side selection dominates") was a latency-regime (bs=1) phenomenon; serving concurrency amortizes the Python selection loop.
4. **Compute is MoE-dominated.** `fused_moe_kernel` = 25.9% of GPU (the 256-expert top-8 layers), the largest single kernel — larger than any single collective.

Net: on NVLink the stock dLLM distributed cost splits roughly **2:1 compute:comm**, with TP all-reduce the only collective worth attacking and the MoE the heaviest compute. The headline distributed lever is exposed TP all-reduce (**32% of every forward step's GPU work**, ~32% of GPU overall) — gates I3 (state-dependent overlap) — but it is far less catastrophic than the A100/no-NVLink picture suggested.

## Setup

### Hardware & software
- **GPUs:** 4× NVIDIA **H100 80GB HBM3**, **full NVLink mesh**. `nvidia-smi topo -m` shows `NV18` (18 bonded NVLink4 links, ~26.6 GB/s each ⇒ ~478 GB/s/dir aggregate) between every GPU pair, so every TP all-reduce / EP all-to-all rides NVLink, not PCIe/`SYS`. Contrast the A100 pass where GPU0/1↔GPU2/3 = `SYS` (cross-socket, slowest hop). NUMA: GPU0 on NUMA0, GPU1–3 on NUMA1, but GPU↔GPU is NVLink regardless, so NUMA only affects host↔device, not collectives.
- **Software:** conda env `sglang` (Python 3.10), SGLang this checkout (git `1464f04b3`), `nsys` 2026.3.1, FlashInfer attention backend. sm90 ⇒ no `a100-sm80-flashinfer-topk-fallback` patch needed (uses the fused topk path directly).
- **Model:** `inclusionAI/LLaDA2.0-mini` — `llada2_moe`, 20 layers, hidden 2048, **256 experts / top-8**, vocab **157184**, bf16, ~17B total / ~1.4B active. KV pool 46.5 GB/GPU (K 23.27 + V 23.27), `max_total_num_tokens=4879616`.

### Parallelism / runtime config (as launched, confirmed from server log)
| Setting | Value | Source |
| --- | --- | --- |
| dLLM algorithm | `LowConfidence`, threshold 0.95 (default) | `--dllm-algorithm LowConfidence` |
| TP / EP | 4 / 4 | `--tp-size 4 --ep-size 4` |
| MoE A2A backend | `none` (fused-Triton MoE + TP all-reduce; no EP all-to-all) | default |
| Attention backend | `flashinfer` (forced by dLLM when graph ON) | dLLM default |
| `mem_fraction_static` | 0.7 | OOM-safe for 256-expert capture |
| `max_running_requests` | 4 | `--max-running-requests 4` |
| `page_size` | 32 (= block_size) | dLLM forced |
| overlap schedule | disabled (`disable_overlap_schedule=True`) | dLLM forced |
| piecewise CUDA graph | disabled | dLLM forced |
| full CUDA graph | ON (`d1_h100_tp4`, bs [1,2,4]) / OFF (`d1_h100_tp4_eager`) | default / `--disable-cuda-graph` |

### Workload
Warmup (outside capture) then **2 concurrent batches of `CONCURRENCY=4`** `/generate` (`max_new_tokens=128`, `temperature=0`). With `--max-running-requests 4`, 4 concurrent requests fill the running batch ⇒ a **GPU-bound** operating point, vs the A100 pass's sequential bs=1 (host-bound). 128 tokens ≈ 4 blocks × 32, each block runs `S_k` denoising steps.

### Runs
| Tag | Knob varied | Purpose |
| --- | --- | --- |
| `d1_h100_tp4` | graph ON + `--cuda-graph-trace=node` | production; faithful in-graph NCCL on NVLink (headline) |
| `d1_h100_tp4_eager` | `--disable-cuda-graph --attention-backend flashinfer` | cross-check / worst-case (eager spin-wait) |

Artifacts: `$DATA_ROOT/profiling/dllm/d1_comm_decomposition/{profiles,logs}/` (`DATA_ROOT` default `/cephfs/shared/wxli/sglang-dllm`): per tag `<tag>.nsys-rep`, `.sqlite`, `<tag>_cuda_gpu_kern_sum.csv`, `<tag>_nvtx_gpu_proj_sum.csv`, `<tag>_nvtx_pushpop_sum.csv`, `<tag>_cuda_gpu_trace.csv`; logs `<tag>_server.log`, `<tag>_stats.log`, `<tag>_summary.txt`.

## Method & tooling

### Instrumentation (reused; minimal, env-gated)
NVTX ranges live in `python/sglang/srt/dllm/profiling.py` (no-op unless `SGLANG_DLLM_NVTX=1`), applied in `dllm/algorithm/low_confidence.py`: `dllm_prefill_forward`, per-step `dllm_forward.step{N}` (wraps `model_runner.forward`) and `dllm_select.step{N}` (host argmax/softmax/threshold/commit), `dllm_final_forward`. The model module `llada2.py` is untouched, so the baseline path is bit-identical with the flag off (CLAUDE.md isolation).

### Capture
`nsys profile -t cuda,nvtx,nccl --cuda-graph-trace=node` wraps the launcher, so all TP/EP child ranks are captured. The window is bounded by SGLang's `CUDA_PROFILER` activity (`cudaProfilerStart/Stop` on the base rank): `/start_profile {"activities":["CUDA_PROFILER"]}` → workload → `/stop_profile`, with `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown`. `--cuda-graph-trace=node` is the crux improvement over the A100 pass: without it the CUDA graph is one opaque node and in-graph NCCL is invisible; with it each in-graph kernel (incl. NCCL) is recorded individually, exposing production collectives **without** disabling the graph (avoiding the eager spin-wait distortion).

### Exact reproduction
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
# headline — production, graph ON, in-graph NCCL traced:
TAG=d1_h100_tp4 bash experiments/profiling/dllm/d1_comm_decomposition/run_d1.sh
# cross-check — eager. run_d1.sh auto-pins flashinfer here: with --disable-cuda-graph
# SGLang no longer force-selects flashinfer and the fa3 fallback crashes on the dLLM
# path (page_table is None, flashattention_backend.py:563).
TAG=d1_h100_tp4_eager EXTRA_ARGS="--disable-cuda-graph" READY_TIMEOUT=600 \
  bash experiments/profiling/dllm/d1_comm_decomposition/run_d1.sh
```

### nsys statistics & post-processing
`parse_d1.py` reads the CSVs from `nsys stats --format csv` with `csv.DictReader` (kernel names contain commas, so naive `split(',')` corrupts columns). Comm vs compute = regex over kernel `Name` (`AllReduce`→TP, `AllToAll`→EP, `AllGather`→LM-head/vocab); a kernel is COMM iff it matches `nccl|all.?reduce|all.?to.?all|all.?gather|…`. Per-phase device cost = `nvtx_gpu_proj_sum` `Total Proj Time` summed per range prefix — **GPU-projected, not CPU push/pop**: push/pop is CPU wall-time that misattributes async graph-launch and `.item()` stalls across phases, so it is reported only as host/wall cost. The **per-forward-step comm proportion** is computed separately and structurally from the `.sqlite` DB (`graph_forward_summary`): for the representative rank (min `deviceId`) it groups in-graph kernels by `graphId`, splits each graph's summed kernel time into collective vs compute, and derives replays as kernels ÷ distinct `graphNodeId`. This is exact for CUDA-graph replays where NVTX-overlap and launch-time projection both fail (see Results). Full canonical method: `notes/experiment_20260619_d1_comm_decomposition.md` §"nsys statistics & post-processing methodology".

## Results
All numbers from the production run `d1_h100_tp4` (graph ON + `--cuda-graph-trace=node`, TP4/EP4, concurrency=4) unless the row says eager. Capture window = 2 concurrent batches of 4 requests × 128 tokens; 1328 forwards (1248 denoising steps + 80 final forwards).

### GPU kernel-time decomposition (`cuda_gpu_kern_sum`, authoritative busy-time)
Total GPU kernel time over the capture = **7137 ms**.
| Class | Time | % of GPU |
| --- | --- | --- |
| **COMM** | **2306 ms** | **32.3%** |
| ↳ `AllReduce(TP)` `…RING_LL` | 2204 ms / 54448 inst (~40.5 µs each) | **30.9%** |
| ↳ `AllGather` (vocab LM-head) | 102 ms / 1328 inst (~77 µs each) | 1.4% |
| ↳ `AllToAll(EP)` | — none (`moe_a2a_backend=none`) | 0% |
| **COMPUTE** | **4831 ms** | **67.7%** |
| ↳ `fused_moe_kernel` | 1849 ms | 25.9% |
| ↳ fused add+RMSNorm | 240 ms | 3.4% |
| ↳ flashinfer paged/prefill attn | 214 + 210 ms | 5.9% |
| ↳ GEMMs (nvjet / cutlass) | 205 + 193 ms | 5.6% |

**Per forward:** 41 all-reduces (20 layers × 2 + 1) → 1.66 ms `AllReduce` + 0.08 ms `AllGather` = **1.74 ms comm/forward**. The 41-per-forward count cross-checks the per-layer structure (attn-out + MoE-out all-reduce each layer, plus the LM-head); `AllGather` at exactly 1328 inst = 1/forward is the vocab-parallel LM-head gather.

### Per-phase device cost (`nvtx_gpu_proj_sum`, GPU-projected per call)
| Phase | GPU/step | CPU-range/step | inst |
| --- | --- | --- | --- |
| `dllm_forward.step` (incl. all collectives) | **7.28 ms** | 5.07 ms | 1248 |
| `dllm_select.step` (host argmax/softmax/commit) | **1.11 ms** | 3.45 ms | 1248 |
| `dllm_final_forward` | 7.96 ms | 5.14 ms | 80 |
| **per-step GPU total (fwd+sel)** | **8.39 ms** | — | — |

Forward = 87% of per-step GPU, select = 13%. Note select CPU-range (3.45 ms) is *below* forward GPU (7.28 ms) → the batched regime is **not** host-bound, the opposite of A100 bs=1 where select CPU was 15.5 ms and dominated wall-clock. (The 7.28 ms projected forward exceeds the 5.03 ms of summed in-graph kernel time per replay because projection includes per-replay launch latency and inter-kernel gaps; for the comm *proportion* within a forward, use the structural split below — dividing comm by the inflated projected time understates it.)

### Per-forward-step comm proportion (structural, by CUDA graph — the accurate per-step number)
A forward step replays a *captured* CUDA graph, so its comm fraction is a fixed property of the kernels **in that graph** — it must be attributed by `graphId`, not by NVTX-window timestamp overlap. Overlap is wrong here: graph replay launches async, so the CPU `dllm_forward.step` window closes *before* its GPU kernels run; time-overlap under-counts forward (~3.5 vs ~7.3 ms/call) and spills comm into `select`. Launch-time projection also fails because in-graph kernels carry the *capture-time* correlation, not the replay's. The `graphId` split is exact. One graph is captured per batch size (all share 654 nodes/replay); rank dev=0, TP-symmetric. The 332 forward-graph replays (312 `forward.step` + 20 `final_forward`) split across three bs:
| Graph | Replays | GPU kernel/step | COMM% | COMP% | Collectives/step |
| --- | --- | --- | --- | --- | --- |
| **5 (dominant bs)** | **251** (76%) | **5.03 ms** | **32.1%** | 67.9% | 42 (41 AllReduce + 1 AllGather) |
| 8 (small bs) | 55 | 4.98 ms | 43.2% | 56.8% | 42 |
| 2 (small bs) | 26 | 6.23 ms | 29.5% | 70.5% | 42 |

**Within one forward step, communication is 32.1% of GPU kernel work at the dominant operating point** (and the all-reduce count is invariant at 42/step = 2/layer × 20 + 1 LM-head AllGather). The fraction rises to 43.2% at the smaller bs because the ~42 collectives stay ~constant (latency-bound RING_LL) while MoE compute scales down with the token count — i.e. comm-per-step is a near-fixed cost the denoising loop pays `S_k`× per block regardless of batch (the D2 amplification lever). Reproduce: the `[graph_forward]` block of `parse_d1.py` (reads `<tag>.sqlite`).

### Eager cross-check (`d1_h100_tp4_eager`, `--disable-cuda-graph --attention-backend flashinfer`)
| Class | Time | % of GPU | vs graph-ON |
| --- | --- | --- | --- |
| Total GPU kernel | 45923 ms | 100% | 6.4× larger |
| **COMM** | **41132 ms** | **89.6%** | comm % 32.3 → 89.6 |
| ↳ `AllReduce(TP)` | 40634 ms / 54120 inst (~751 µs each) | 88.5% | per-call ~40 → ~751 µs (18.5×) |
| ↳ `AllGather` | 498 ms / 1320 inst (~377 µs each) | 1.1% | per-call ~77 → ~377 µs |
| COMPUTE | 4791 ms | 10.4% | ~unchanged (fused_moe 1842 vs 1849 ms) |

Per-phase: eager `dllm_forward.step` = 58.08 ms GPU (vs 7.28 graph-ON, ~8×), forward 98% / select 2%. **Compute is identical; only comm exploded.** The decisive finding: eager `AllReduce` here is ~751 µs/call — essentially the same as the A100/`SYS` eager run (~700 µs/call) despite NVLink being ~10× the bandwidth. So the eager inflation is **not** a transfer-bandwidth or cross-socket effect; it is **cross-rank launch-desync spin-wait** (eager launches each collective separately, CPU-launch jitter desynchronizes the 4 ranks, and every `RING_LL` all-reduce spins waiting for the slowest peer). The CUDA graph removes it on both topologies by replaying the forward in lockstep so peers arrive together (graph-ON all-reduce = 40 µs). The A100 "87% comm" headline reproduces here as **89.6% comm on NVLink**, confirming it is an eager-mode artifact, interconnect-independent, and *not* a production number. Production exposed comm is the **32.3%** graph-ON figure.

### A100 (no NVLink) vs H100 (NV18) — same experiment, different interconnect
| Quantity | A100 PCIe/`SYS`, bs=1 | H100 NV18, conc=4 |
| --- | --- | --- |
| operating regime | host-bound (GPU idle ~92%) | GPU/forward-bound |
| production `AllReduce`/call | hidden in graph (uncaptured) | ~40 µs (captured via node-trace) |
| eager `AllReduce`/call | ~0.70 ms (spin-wait across `SYS`) | ~0.75 ms (spin-wait, ~same) |
| vocab `AllGather`/call | ~5.6 ms (157k vocab over `SYS`) | ~77 µs (graph) / 0.38 ms (eager) |
| comm % of GPU | "87%" (eager artifact only) | 32.3% (faithful, graph ON) |
| dominant cost | host-side selection + full-vocab logits | TP all-reduce + MoE compute |

## Caveats (read before generalizing)
- **NVLink-specific:** every collective rides the NV18 mesh; numbers do not transfer to the A100/PCIe/`SYS` topology (see the legacy report) or to inter-node NCCL.
- **Eager (`--disable-cuda-graph`) inflates collective cost via cross-rank spin-wait** — a worst-case bound, not a production comm number. Production comm is the graph-ON + `--cuda-graph-trace=node` measurement. Eager also needs `--attention-backend flashinfer` pinned or the fa3 fallback crashes on the dLLM path (`run_d1.sh` auto-pins it).
- **`moe_a2a_backend='none'`** ⇒ no EP all-to-all in this run; EP-dispatch is D4 (`--moe-a2a-backend deepep`), which would trade the TP all-reduce for an EP all-to-all.
- **Single base-rank capture** → one representative TP rank; MoE expert routing is the only genuinely rank-divergent behavior (D4).
- **Operating point is concurrency=4 (GPU-bound)**; the compute:comm split and the host-bound conclusion both shift with batch (→ concurrency sweep).

## Takeaways for direction priority
Mapping to `notes/dllm_distributed_optimization_directions.md`:
- **TP all-reduce is the one comm lever (I3 — state-dependent overlap).** ~24% of every forward, fully exposed because dLLM force-disables overlap. On NVLink it is ~1.66 ms/forward of real transfer/launch (not spin), so overlapping or fusing it is worth ~20% of forward time — meaningful but bounded. The `S_k`× amplification (D2) multiplies this across denoising steps, so step-reduction (Design-1) attacks the same cost from the other side.
- **MoE compute (25.9%) now rivals all collectives combined.** The 256-expert `fused_moe_kernel` is the heaviest single kernel; MoE-runner / expert-efficiency work is competitive with comm work on this interconnect. This brings D4 forward: measure whether the EP all-to-all (`--moe-a2a-backend deepep`) is cheaper than the TP all-reduce it replaces on NVLink.
- **De-prioritize host-side selection (E-a) for the serving regime.** It dominated at A100 bs=1 but is only 13% of per-step GPU at concurrency=4. Still relevant for latency-bound single-stream dLLM, not for throughput serving.
- **De-prioritize full-vocab `AllGather` (E-b) on NVLink.** 1.4% of GPU here vs the 5.6 ms/call it cost over A100 `SYS`. The full-vocab softmax in selection is still a host/compute cost, but the collective side is no longer a target on NVLink.
- **Engineering vs innovation:** CUDA-graphing the static-shape forward is already ON and is what keeps collectives cheap/in-lockstep — that win is spent. The remaining distributed headroom is overlap (blocked by the loop mutating `input_ids` mid-step → D6) and step-count reduction — both innovation-side.

## Next
- Concurrency sweep (1/2/4/8) graph-ON: locate the host-bound→GPU-bound crossover and watch the comm fraction grow with batch.
- D2: combine `comm_per_step` here with `S_k` (the §3 counter in the plan) for the comm-per-output-token `S_k`× penalty vs AR.
- D4: relaunch with `--moe-a2a-backend deepep` to bring EP all-to-all into the decomposition and check expert-load drift across denoising steps.
- Scripts: this dir (`run_d1.sh`, `parse_d1.py`). Data: `$DATA_ROOT/profiling/dllm/d1_comm_decomposition/`.
