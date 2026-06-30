# dLLM Distributed-Efficiency Profiling

Experiment **scripts** for the dLLM distributed-efficiency profiling plan
(`notes/dllm_baseline_profiling_plan.md`, experiments **D1–D9**). Scripts live in
the repo; **outputs (profiles, logs, CSVs) are data and go to `/cephfs`** in a
tree that *mirrors* this hierarchy so scripts and data line up 1:1.

## Layout (one folder per experiment)

```
repo : experiments/profiling/dllm/
         README.md                       # this file
         common/                         # shared helpers (if any)
         d1_comm_decomposition/          # D1 — per-step comm decomposition & exposed fraction
           run_d1.sh                     # launch under nsys, bound capture, dump CSVs
           parse_d1.py                   # comm-vs-compute + per-step NVTX summary
           README.md                     # the D1 experiment report

data : $DATA_ROOT/profiling/dllm/<exp>/  (DATA_ROOT default /cephfs/shared/wxli/sglang-dllm)
           profiles/                     # <TAG>.nsys-rep, .sqlite, *_cuda_gpu_kern_sum.csv, ...
           logs/                         # <TAG>_server.log, <TAG>_stats.log, <TAG>_summary.txt
```

Override `DATA_ROOT` (whole tree) or `OUT` (one experiment) to relocate.

## Plan → folder map

| Exp | Folder | Distributed question |
|---|---|---|
| **D1** | `d1_comm_decomposition/` | exposed-comm fraction & TP/EP/DP split per denoising step |
| **D2** | `d2_sk_amplification/h100/` | comm-per-output-token vs AR (`S_k`× penalty), 4×H100 NVLink |
| **D2** | `d2_sk_amplification/a100/` | same on 4×A100 (+bs=1): comm projected from volume onto A100 NVLink vs measured A100 compute (comm fraction) |
| D3 | _(todo)_ | strong scaling across TP shapes |
| D4 | _(todo)_ | EP all-to-all volume & expert-load drift across steps |
| D5 | _(todo)_ | DP-attention gather/scatter cost in the loop |
| D6 | _(todo)_ | exposed-comm / overlap headroom |
| D7 | _(todo)_ | wasted collective rounds from stragglers |
| D8 | _(todo)_ | per-rank KV/MoE residency, page granularity |
| D9 | _(todo)_ | inter-node collective cost (multi-node) |

### FOCUS experiments (algorithmic compute reduction, not the D-series distributed plan)

| Exp | Folder | Question |
|---|---|---|
| **F1** | `focus_vs_lowconf/` | FOCUS paper-exact reduced forward vs LowConfidence — throughput/latency + processed-token redundancy Σ\|S\|/(B·bs), LLaDA2.0-mini 1×A100 TP=1 eager |

## Prerequisites

- conda env `sglang` active; `nsys` on PATH.
- NVTX ranges enabled via `SGLANG_DLLM_NVTX=1` (D1, set by the scripts) and the
  per-block `S_k` step counter via `SGLANG_DLLM_PROFILE=1` (D2) — both implemented
  in `python/sglang/srt/dllm/profiling.py`, used by
  `python/sglang/srt/dllm/algorithm/low_confidence.py`. No-op when unset, so the
  baseline path is bit-identical with the flags off.
- Working launch config: see memory `llada2-launch-config-a100` and
  `dllm-nsys-profiling-method`; on A100 (sm80) the `a100-sm80-flashinfer-topk-fallback`
  patch is required (H100/sm90 uses the fused path directly).

## Key gotcha (carries across all comm experiments)

The dLLM forward is **CUDA-graphed by default**, so NCCL kernels are *inside* the
graph and invisible to plain nsys. Two ways to see them:
- **Preferred:** keep the graph and add `nsys --cuda-graph-trace=node` so in-graph
  kernels (incl. NCCL) are recorded individually — faithful production comm.
- **Cross-check only:** `--disable-cuda-graph` (eager) exposes NCCL but *inflates*
  it via cross-rank spin-wait — a worst-case bound, not a production number.

`run_d1.sh` does the preferred method for the production run and supports the
eager run via `EXTRA_ARGS="--disable-cuda-graph"`.
