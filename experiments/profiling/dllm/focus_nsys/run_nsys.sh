#!/usr/bin/env bash
# F6 — nsys kernel-level bottleneck decomposition of kernel-FOCUS vs LowConfidence
# (LLaDA2.0-mini, 1xA100 80GB, TP=1, eager). Explains WHY the FOCUS speedup is
# minor (~1.05x vs LowConfidence, far from the paper's 2.32x).
#
# Method (see memory dllm-nsys-profiling-method + nsys-2025-bundled-no-nccl-graphid):
#   - launch the server under nsys, bracket the capture with SGLang's
#     /start_profile CUDA_PROFILER (cudaProfilerStart/Stop on the base rank),
#     drive a batched workload, /stop_profile.
#   - SGLANG_DLLM_NVTX=1 tags the FOCUS phases (focus_prefix / focus_l1_attn /
#     focus_suffix / dllm_focus_commit / dllm_focus_final_forward) and the
#     LowConfidence steps (dllm_forward.stepN / dllm_select.stepN).
#   - single GPU => no NCCL; -t cuda,nvtx (this box's nsys has no nccl plugin).
#   - eager (--disable-cuda-graph): FOCUS is eager anyway; this also lets nsys see
#     every per-layer kernel launch so we can measure GPU-idle (launch-gap) time.
#
# Usage:  ./run_nsys.sh focus     # kernel-FOCUS (SGLANG_FOCUS_KERNEL=1)
#         ./run_nsys.sh lowconf   # LowConfidence baseline
#         ./run_nsys.sh           # both
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/focus_nsys}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
PROF=${PROF:-$OUT/profiles}
LOGS=${LOGS:-$OUT/logs}
FVL="$REPO/experiments/profiling/dllm/focus_vs_lowconf"
NSYS=${NSYS:-/root/miniconda3/envs/sglang/nsight-compute-2025.2.1/host/target-linux-x64/nsys}
PORT=${PORT:-30032}
TP=${TP:-1}
MODEL=${MODEL:-/cephfs/shared/model/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC=${CONC:-8}
GEN_TOKENS=${GEN_TOKENS:-128}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HOST=127.0.0.1

mkdir -p "$PROF" "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HOME=${HF_HOME:-/cephfs/shared/huggingface}
export SGLANG_DLLM_NVTX=1

WHICH=${1:-both}

kill_servers() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null
  local t=0
  while :; do
    local mem procs
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    procs=$(pgrep -fc "sglang.launch_server" || echo 0)
    if [ "${mem:-9999}" -lt 2000 ] && [ "${procs:-0}" -eq 0 ]; then break; fi
    sleep 3; t=$((t+3)); [ "$t" -ge 120 ] && break
  done
}

PROMPTS=(
  "Write a Python function that returns the n-th Fibonacci number, with a short explanation."
  "Q: What is 17*23? Think step by step and show the multiplication.\nA:"
  "Explain how a binary search works and give a short Python implementation."
  "Q: A train travels 60 km in 45 minutes. What is its average speed in km/h? Reason step by step.\nA:"
)
fire_batch() {
  local pids=()
  for ((i=0; i<CONC; i++)); do
    curl -s --noproxy '*' "http://$HOST:$PORT/generate" -H 'Content-Type: application/json' \
      -d "{\"text\":\"${PROMPTS[$((i % ${#PROMPTS[@]}))]}\",\"sampling_params\":{\"max_new_tokens\":$GEN_TOKENS,\"temperature\":0}}" >/dev/null &
    pids+=($!)
  done
  wait "${pids[@]}"
}

run_one() {  # $1 = focus|lowconf
  local MODE=$1 ALGO CFG TAG
  if [ "$MODE" = "focus" ]; then
    ALGO=Focus; CFG="$FVL/configs/focus.yaml"; TAG="focus_kernel_c${CONC}"
    export SGLANG_FOCUS_KERNEL=1
  else
    ALGO=LowConfidence; CFG="$FVL/configs/low_confidence.yaml"; TAG="lowconf_c${CONC}"
    unset SGLANG_FOCUS_KERNEL
  fi
  local REP="$PROF/$TAG"
  local SRVLOG="$LOGS/${TAG}_server.log"
  echo "=============================================================="
  echo "[f6] $MODE algo=$ALGO conc=$CONC -> ${REP}.nsys-rep"
  kill_servers

  "$NSYS" profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    --force-overwrite true -o "$REP" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code \
      --dllm-algorithm "$ALGO" --dllm-algorithm-config "$CFG" \
      --host 0.0.0.0 --port "$PORT" --tp-size "$TP" \
      --mem-fraction-static "$MEMFRAC" --max-running-requests "$CONC" \
      --disable-cuda-graph --attention-backend flashinfer \
      > "$SRVLOG" 2>&1 &
  local NSYS_PID=$!
  cleanup() { kill "$NSYS_PID" 2>/dev/null; kill_servers; }
  trap cleanup EXIT

  echo "[f6] waiting for server (timeout ${READY_TIMEOUT}s)..."
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$NSYS_PID" 2>/dev/null; then echo "[f6] died:"; tail -40 "$SRVLOG"; trap - EXIT; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[f6] TIMEOUT"; cleanup; trap - EXIT; return 1; fi
  done
  echo "[f6] ready after ${t}s"

  echo "[f6] warmup (not captured)..."; fire_batch
  echo "[f6] /start_profile"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/start_profile" \
    -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}' >/dev/null
  echo "[f6] captured workload (1 batch of $CONC)..."; fire_batch
  echo "[f6] /stop_profile"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/stop_profile" >/dev/null 2>&1 || true

  echo "[f6] waiting for nsys to finalize..."
  wait "$NSYS_PID" 2>/dev/null
  trap - EXIT
  if [ ! -f "${REP}.nsys-rep" ]; then echo "[f6] ERROR no report"; tail -40 "$SRVLOG"; return 1; fi

  echo "[f6] nsys stats..."
  "$NSYS" stats --force-export=true \
    --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,cuda_gpu_trace \
    --format csv --output "$REP" "${REP}.nsys-rep" \
    > "$LOGS/${TAG}_stats.log" 2>&1
  echo "[f6] $MODE done: $(ls ${REP}*_cuda_gpu_kern_sum.csv 2>/dev/null)"
  kill_servers
}

case "$WHICH" in
  focus)   run_one focus ;;
  lowconf) run_one lowconf ;;
  both)    run_one focus; run_one lowconf ;;
  *) echo "usage: $0 [focus|lowconf|both]"; exit 2 ;;
esac

echo "[f6] parse:"
python "$REPO/experiments/$EXP_PATH/parse_nsys.py" "$PROF" 2>&1 | tee "$LOGS/focus_nsys_summary.txt" || true
echo "[f6] ALL DONE"
