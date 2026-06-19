<!--
Experiment report template for dLLM-on-SGLang profiling/benchmark experiments.
Copy to notes/experiment_YYYYMMDD_<short_name>.md and fill in. Delete guidance
comments. Keep it dense (CLAUDE.md style): clear headings, compact bullets, no
filler. Scripts live in-repo under experiments/<family>/<subject>/; data/outputs
go to $DATA_ROOT/<family>/<subject>/ (mirrors the repo tree; DATA_ROOT default
/cephfs/shared/wxli/sglang-dllm).
-->

# Experiment <ID> — <Title> (<model>, <HW e.g. 4×A100>)

## Summary

<2–4 sentences / numbered findings. Lead with the conclusion and the headline
number(s). Flag anything that corrects a prior result. State the operating point
(batch size / concurrency) since conclusions depend on it.>

## Setup

### Hardware & software

- **GPUs:** <count × model>, **interconnect** (NVLink/NVSwitch vs PCIe). Paste the
  relevant `nvidia-smi topo -m` lines if topology affects comm; note NVLink status
  (`nvidia-smi nvlink --status`). Topology is first-class for any comm result.
- **Software:** conda env, Python, SGLang commit (`git rev-parse HEAD`), nsys/torch
  versions, key libs (FlashInfer, NCCL).
- **Model:** name, arch/params (layers, hidden, experts/top-k, vocab), dtype, per-GPU footprint.

### Parallelism / runtime config (as launched)

| Setting | Value | Source (flag / forced) |
| --- | --- | --- |
| dLLM algorithm | <e.g. LowConfidence, threshold> | `--dllm-algorithm …` |
| TP / EP / PP / DP | … | `--tp-size …` |
| MoE A2A backend | … | `--moe-a2a-backend …` |
| CUDA graph | ON / OFF | default / `--disable-cuda-graph` |
| overlap schedule | … | dLLM forced? |
| mem_fraction_static, max_running_requests, page_size | … | … |

<Confirm the above against the server log (paste the decisive lines). Note any
patches required (e.g. [[a100-sm80-flashinfer-topk-fallback]]).>

### Workload

- Warmup (outside capture) + profiled workload: prompts, `max_new_tokens`,
  temperature, **concurrency / effective batch size**, request count.
- State the regime this implies (latency-bound bs=1 vs GPU-bound batched).

### Runs

| Tag | Knob varied | Purpose |
| --- | --- | --- |
| … | … | … |

Artifacts: `DATA_ROOT/<family>/<subject>/{profiles,logs}/` (mirrors repo
`experiments/<family>/<subject>/`).

## Method & tooling

### Instrumentation added (keep minimal; isolate from production path)

- <new files / NVTX ranges / counters; env-gate them (e.g. SGLANG_DLLM_NVTX=1) so
  the baseline path is bit-identical when off. Note what was *reused* vs *added*.>

### Capture

- <How the capture window is bounded — e.g. nsys `--capture-range=cudaProfilerApi`
  driven by `/start_profile {"activities":["CUDA_PROFILER"]}` → workload →
  `/stop_profile`. For comm with graph ON, add `--cuda-graph-trace=node`.>

### Exact reproduction

```bash
# env + driver invocation(s); copy-pasteable
```

### nsys statistics & post-processing methodology

<How raw nsys output becomes the reported numbers. Reuse/adapt the canonical
description — see notes/experiment_20260619_d1_comm_decomposition.md §"nsys
statistics & post-processing methodology" and experiments/profiling/dllm/parse_d1.py.
At minimum state:>

- **Reports generated** (`nsys stats --report … --format csv`): which CSVs and what each gives.
- **CSV parsing:** use `csv.DictReader` (kernel names contain commas); sqlite as source of truth.
- **Comm vs compute:** regex over kernel `Name`; group collectives by op (AllReduce=TP, AllToAll=EP, AllGather=LM-head/vocab).
- **Per-phase device cost:** aggregate `nvtx_gpu_proj_sum` `Total Proj Time` per range prefix — **GPU-projected, not CPU push/pop** (push/pop misattributes async/`.item()` stalls; report it only as wall/host cost).
- **Comm fraction of E2E:** `Σ NCCL GPU time ÷ E2E time`, measured **graph ON** with `--cuda-graph-trace=node`, at a **GPU-bound** operating point (bs=1 is host-bound → unrepresentative).
- **Sanity checks:** derived counts (e.g. forwards ≈ AllReduce_inst / (layers×collectives)); cross-check phase sums vs kernel-sum total.

## Results

<Tables first, prose second. Always label the metric (GPU-projected vs CPU-wall,
% of GPU vs % of E2E). State the operating point with every number.>

## Caveats (read before generalizing)

<Be explicit about what would change the result: interconnect/topology, batch
size, CUDA-graph on/off (eager inflates collectives via spin-wait — not a faithful
comm proxy), single-rank capture, MoE backend, spin-wait in NCCL kernel time, etc.>

## Takeaways for direction priority

<Map findings to the optimization directions (notes/dllm_distributed_optimization_directions.md):
which direction each result promotes/demotes, and why. Distinguish engineering vs
innovation per the project's framing.>

## Next

- <follow-up experiments, with the specific question each answers>
- Scripts: `experiments/<family>/<subject>/…`. Data: `DATA_ROOT/<family>/<subject>/…`.
