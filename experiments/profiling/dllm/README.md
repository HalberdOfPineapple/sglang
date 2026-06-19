# dLLM Profiling Experiments

Experiment **scripts** for the dLLM distributed-efficiency profiling plan
(`notes/dllm_baseline_profiling_plan.md`). Scripts live in the repo; **outputs
(profiles, logs, CSVs) are data and go to `/cephfs`**, in a tree that **mirrors
this `experiments/` hierarchy** so scripts and data line up 1:1.

## Layout (scripts mirror data)

```
repo : experiments/profiling/dllm/
         run_d1.sh      # D1: per-step comm decomposition & exposed-comm fraction
         parse_d1.py    # summarize nsys CSVs (comm-vs-compute + per-step NVTX)

data : $DATA_ROOT/profiling/dllm/          (DATA_ROOT default /cephfs/shared/wxli/sglang-dllm)
         profiles/      # <TAG>.nsys-rep, .sqlite, *_cuda_gpu_kern_sum.csv, *_nvtx_pushpop_sum.csv, ...
         logs/          # <TAG>_server.log, <TAG>_stats.log
```

Override `DATA_ROOT` (whole tree) or `OUT` (this experiment only) to relocate.

## Prerequisites

- conda env `sglang` active; `nsys` on PATH.
- NVTX ranges enabled via `SGLANG_DLLM_NVTX=1` (set by the scripts) — implemented
  in `python/sglang/srt/dllm/profiling.py`, used by
  `python/sglang/srt/dllm/algorithm/low_confidence.py`. No-op when unset.
- Working launch config: see memory `llada2-launch-config-a100` and
  `dllm-nsys-profiling-method`.

## Usage

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
# Run A — production (CUDA graph ON)
TAG=d1_tp4        bash experiments/profiling/dllm/run_d1.sh
# Run B — eager (exposes graph-internal NCCL; CUDA graph hides it)
TAG=d1_tp4_eager EXTRA_ARGS="--disable-cuda-graph" READY_TIMEOUT=600 \
                  bash experiments/profiling/dllm/run_d1.sh
```

Override knobs via env: `TP`, `EP`, `MEMFRAC`, `MAXREQ`, `GEN_TOKENS`, `MODEL`,
`PORT`, `OUT`, `EXTRA_ARGS`.

## Key gotcha

The dLLM forward is CUDA-graphed by default, so NCCL kernels are *inside* the
graph and invisible to nsys — use `--disable-cuda-graph` (Run B) to decompose
comm. Findings: `notes/experiment_20260619_d1_comm_decomposition.md`.
