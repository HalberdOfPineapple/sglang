#!/usr/bin/env bash
# D1 — per-step comm decomposition & exposed-comm fraction for the stock
# LowConfidence dLLM path (see notes/dllm_baseline_profiling_plan.md, D1, and
# notes/experiment_20260619_d1_comm_decomposition.md).
#
# Launches the server under nsys, brackets the capture with SGLang's
# /start_profile CUDA_PROFILER activity (cudaProfilerStart/Stop on the base
# rank), drives a small denoising workload, then dumps nsys stats to CSV.
#
# Reuses: SGLang profiler mixin (CUDA_PROFILER -> cudaProfilerStart) + NVTX ranges
# added in dllm/profiling.py (SGLANG_DLLM_NVTX=1). nsys auto-tags NCCL kernels.
#
# Scripts live in the repo (this dir); OUTPUTS (profiles/logs = data) go to the
# data root, which MIRRORS the repo experiments/ hierarchy:
#   repo : experiments/profiling/dllm/
#   data : $DATA_ROOT/profiling/dllm/{profiles,logs}/
# Override DATA_ROOT (or OUT) to relocate data.
set -uo pipefail

# ---- config (override via env) -------------------------------------------------
REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}   # data lives under here
EXP_PATH=${EXP_PATH:-profiling/dllm}                      # mirrors experiments/profiling/dllm
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
PROF=${PROF:-$OUT/profiles}
LOGS=${LOGS:-$OUT/logs}
TAG=${TAG:-d1_tp4}
PORT=${PORT:-30000}
TP=${TP:-4}; EP=${EP:-4}
MODEL=${MODEL:-inclusionAI/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
MAXREQ=${MAXREQ:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}        # e.g. "--disable-cuda-graph" to expose graph-internal NCCL
GEN_TOKENS=${GEN_TOKENS:-128}        # ~4 blocks of 32 -> enough denoising steps
READY_TIMEOUT=${READY_TIMEOUT:-2400} # allow first-time model download
HOST=127.0.0.1
REP=$PROF/${TAG}

mkdir -p "$PROF" "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export SGLANG_DLLM_NVTX=1            # enable the dLLM NVTX ranges
export SGLANG_TORCH_PROFILER_DIR=$PROF
SRVLOG=$LOGS/${TAG}_server.log

echo "[d1] report -> ${REP}.nsys-rep ; server log -> $SRVLOG"

# ---- launch server under nsys (background) ------------------------------------
# capture-range=cudaProfilerApi: nsys records only between cudaProfilerStart/Stop.
# stop-shutdown: end collection AND shut the server down after /stop_profile.
nsys profile \
  -t cuda,nvtx,nccl \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite true \
  -o "$REP" \
  python -m sglang.launch_server \
    --model-path "$MODEL" --dllm-algorithm LowConfidence \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tp-size "$TP" --ep-size "$EP" \
    --mem-fraction-static "$MEMFRAC" --max-running-requests "$MAXREQ" \
    $EXTRA_ARGS \
    > "$SRVLOG" 2>&1 &
NSYS_PID=$!
echo "[d1] nsys+server pid=$NSYS_PID"

cleanup() { kill "$NSYS_PID" 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; }
trap cleanup EXIT

# ---- wait until model is loaded ------------------------------------------------
echo "[d1] waiting for server (timeout ${READY_TIMEOUT}s; first run downloads weights)..."
t=0
until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
  sleep 5; t=$((t+5))
  if ! kill -0 "$NSYS_PID" 2>/dev/null; then echo "[d1] server process died; tail log:"; tail -40 "$SRVLOG"; exit 1; fi
  if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[d1] TIMEOUT waiting for server"; tail -40 "$SRVLOG"; exit 1; fi
done
echo "[d1] server ready after ${t}s"

gen() {  # $1 = prompt
  curl -s --noproxy '*' "http://$HOST:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"$1\",\"sampling_params\":{\"max_new_tokens\":$GEN_TOKENS,\"temperature\":0}}" >/dev/null
}

# ---- warmup (NOT captured: capture starts only at cudaProfilerStart) ----------
echo "[d1] warmup..."
gen "Q: What is 17*23? Think step by step.\nA:"

# ---- start capture, run profiled workload, stop -------------------------------
echo "[d1] /start_profile (CUDA_PROFILER) -> nsys capture begins"
curl -s --noproxy '*' -X POST "http://$HOST:$PORT/start_profile" \
  -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}' >/dev/null

echo "[d1] profiled generation..."
gen "Q: A train travels 60 km in 45 minutes. What is its average speed in km/h? Reason step by step.\nA:"
gen "Write a Python function that returns the n-th Fibonacci number, with a short explanation."

echo "[d1] /stop_profile -> nsys stops + shuts down server"
curl -s --noproxy '*' -X POST "http://$HOST:$PORT/stop_profile" >/dev/null 2>&1 || true

# ---- wait for nsys to finalize the report -------------------------------------
echo "[d1] waiting for nsys to write report..."
wait "$NSYS_PID" 2>/dev/null
trap - EXIT
if [ ! -f "${REP}.nsys-rep" ]; then echo "[d1] ERROR: ${REP}.nsys-rep not found"; tail -40 "$SRVLOG"; exit 1; fi
echo "[d1] report written: ${REP}.nsys-rep"

# ---- post-process: NCCL/kernel + NVTX per-step breakdown ----------------------
echo "[d1] nsys stats (this can take a minute)..."
# nvtx_gpu_proj_sum = GPU time PROJECTED onto each NVTX range (the meaningful
# per-phase metric). nvtx_pushpop_sum = CPU push/pop wall-time (misleading on its
# own: async graph launch + lazy .item() syncs misattribute work across phases).
nsys stats --force-export=true \
  --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,nvtx_pushpop_sum,cuda_gpu_trace \
  --format csv --output "$REP" "${REP}.nsys-rep" \
  > "$LOGS/${TAG}_stats.log" 2>&1
echo "[d1] stats CSVs:"; ls -1 ${REP}*_cuda_gpu_kern_sum.csv ${REP}*_nvtx_pushpop_sum.csv 2>/dev/null

# ---- summarize (comm vs compute + per-step NVTX) ------------------------------
python "$REPO/experiments/profiling/dllm/parse_d1.py" "$REP" || true
echo "[d1] DONE"
