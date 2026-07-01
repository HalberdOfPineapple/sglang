#!/usr/bin/env bash
# Experiment F5 — FOCUS Phase-B Triton kernels (§B1 importance + §B2 selection)
# ON vs OFF (LLaDA2.0-mini, 1xA100 80GB, TP=1, eager).
#
# Answers: do the §B kernels (SGLANG_FOCUS_KERNEL=1) reduce the host-bound
# per-step cost the F3-lite timing attributed to `select ~14%` (Python
# per-request selection loop) + the L0/L1 importance einsum inside `p_fwd`?
#
# Two measurement modes (kept SEPARATE — phase timing inserts CUDA syncs at
# phase boundaries that serialize and distort tok/s):
#   (T) throughput : kernel {0,1} x conc {1,8,16}, NO phase timing -> tok/s.
#   (P) phase split: kernel {0,1} at conc 8, SGLANG_FOCUS_PHASE_TIMING=1 ->
#                    aggregate the per-block `[focus-timing]` shares.
# Both are FOCUS-only (α=1.5). Redundancy is logged in (T) to confirm the kernel
# path evicts identically to the oracle (same Σ|S|/(B·bs) distribution).
#
# Scripts in repo (this dir); reuses the focus_vs_lowconf driver + configs.
# OUTPUTS (data) mirror to $OUT/logs.
#
# Usage:  ./run_focus_kernel.sh                 # full: T sweep + P at conc 8
#         CONC_LIST="1 8" ./run_focus_kernel.sh
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/focus_kernel}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
LOGS=${LOGS:-$OUT/logs}
FVL="$REPO/experiments/profiling/dllm/focus_vs_lowconf"   # reuse driver+config
HERE="$REPO/experiments/$EXP_PATH"
PORT=${PORT:-30031}
TP=${TP:-1}
MODEL=${MODEL:-/cephfs/shared/model/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC_LIST=${CONC_LIST:-"1 8 16"}
PHASE_CONC=${PHASE_CONC:-8}
HUMANEVAL=${HUMANEVAL:-/cephfs/shared/wxli/human-eval/data/HumanEval.jsonl.gz}
N_SAMPLES=${N_SAMPLES:-20}
MAX_NEW=${MAX_NEW:-128}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HOST=127.0.0.1

mkdir -p "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HOME=${HF_HOME:-/cephfs/shared/huggingface}

echo "[fk] repo=$REPO data=$OUT model=$MODEL TP=$TP conc='$CONC_LIST'"
echo "[fk] commit=$(cd "$REPO" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

kill_servers() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "sglang::" 2>/dev/null
  local t=0
  while :; do
    local mem procs
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    procs=$(pgrep -fc "sglang.launch_server" || echo 0)
    if [ "${mem:-9999}" -lt 2000 ] && [ "${procs:-0}" -eq 0 ]; then break; fi
    sleep 3; t=$((t+3)); [ "$t" -ge 120 ] && break
  done
}

# $1=KERNEL(0|1)  $2=conc  $3=mode(T|P)
run_one() {
  local KERNEL=$1 CONC=$2 MODE=$3
  local TAG="k${KERNEL}_c${CONC}_${MODE}"
  local SRVLOG="$LOGS/${TAG}_server.log"
  local DRIVELOG="$LOGS/${TAG}_drive.log"
  local RESULT="$LOGS/${TAG}_result.json"
  local REDCSV="$LOGS/${TAG}_redundancy.csv"
  local TOTAL=$(( CONC * 3 )); [ "$TOTAL" -lt 6 ] && TOTAL=6

  echo "=============================================================="
  echo "[fk] KERNEL=$KERNEL conc=$CONC mode=$MODE total_reqs=$TOTAL -> $RESULT"

  export SGLANG_FOCUS_KERNEL="$KERNEL"
  export SGLANG_FOCUS_LOG_REDUNDANCY=1 SGLANG_FOCUS_REDUNDANCY_CSV="$REDCSV"; rm -f "$REDCSV"
  if [ "$MODE" = "P" ]; then export SGLANG_FOCUS_PHASE_TIMING=1; else unset SGLANG_FOCUS_PHASE_TIMING; fi

  kill_servers
  PYTHONUNBUFFERED=1 python -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --dllm-algorithm Focus --dllm-algorithm-config "$FVL/configs/focus.yaml" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP" --mem-fraction-static "$MEMFRAC" \
    --max-running-requests "$CONC" \
    --disable-cuda-graph --attention-backend flashinfer \
    > "$SRVLOG" 2>&1 &
  local SRV_PID=$!
  cleanup() { kill "$SRV_PID" 2>/dev/null; kill_servers; }
  trap cleanup EXIT

  echo "[fk] waiting for server (timeout ${READY_TIMEOUT}s)..."
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[fk] server died:"; tail -40 "$SRVLOG"; trap - EXIT; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[fk] TIMEOUT"; tail -40 "$SRVLOG"; cleanup; trap - EXIT; return 1; fi
  done
  echo "[fk] server ready after ${t}s"
  sleep 3

  # warmup (unmeasured)
  python "$FVL/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$CONC" \
    "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" >/dev/null 2>&1 || true

  echo "[fk] measured workload (conc=$CONC, $TOTAL reqs, mode=$MODE)..."
  python "$FVL/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$TOTAL" \
    "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" "$RESULT" 2>&1 | tee "$DRIVELOG"

  echo "[fk] redundancy rows: $(( $(wc -l < "$REDCSV" 2>/dev/null || echo 1) - 1 ))"
  echo "[fk] focus-timing lines: $(grep -c 'focus-timing' "$SRVLOG" 2>/dev/null || echo 0)"
  cleanup; trap - EXIT
}

# (T) throughput sweep — no phase timing
for c in $CONC_LIST; do
  run_one 0 "$c" T
  run_one 1 "$c" T
done

# (P) phase split at PHASE_CONC — timing ON
run_one 0 "$PHASE_CONC" P
run_one 1 "$PHASE_CONC" P

echo "[fk] ALL DONE — parse:"
python "$HERE/parse_focus_kernel.py" "$LOGS" 2>&1 | tee "$LOGS/focus_kernel_summary.txt" || true
