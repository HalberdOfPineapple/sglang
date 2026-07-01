#!/usr/bin/env bash
# Same-session LowConfidence baseline for the F5 focus_kernel comparison.
# Identical harness to run_focus_kernel.sh (T mode: no phase timing) so
# LowConfidence tok/s is directly comparable to the FOCUS OFF/ON numbers.
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/focus_kernel}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
LOGS=${LOGS:-$OUT/logs}
FVL="$REPO/experiments/profiling/dllm/focus_vs_lowconf"
PORT=${PORT:-30031}
TP=${TP:-1}
MODEL=${MODEL:-/cephfs/shared/model/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC_LIST=${CONC_LIST:-"1 8 16"}
HUMANEVAL=${HUMANEVAL:-/cephfs/shared/wxli/human-eval/data/HumanEval.jsonl.gz}
N_SAMPLES=${N_SAMPLES:-20}
MAX_NEW=${MAX_NEW:-128}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HOST=127.0.0.1

mkdir -p "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HOME=${HF_HOME:-/cephfs/shared/huggingface}
unset SGLANG_FOCUS_KERNEL SGLANG_FOCUS_PHASE_TIMING SGLANG_FOCUS_LOG_REDUNDANCY SGLANG_FOCUS_REDUNDANCY_CSV

echo "[lc] commit=$(cd "$REPO" && git rev-parse --short HEAD)  conc='$CONC_LIST'"

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

run_one() {
  local CONC=$1
  local TAG="lowconf_c${CONC}_T"
  local SRVLOG="$LOGS/${TAG}_server.log" RESULT="$LOGS/${TAG}_result.json"
  local TOTAL=$(( CONC * 3 )); [ "$TOTAL" -lt 6 ] && TOTAL=6
  echo "=== LowConfidence conc=$CONC total=$TOTAL -> $RESULT"
  kill_servers
  PYTHONUNBUFFERED=1 python -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --dllm-algorithm LowConfidence --dllm-algorithm-config "$FVL/configs/low_confidence.yaml" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP" --mem-fraction-static "$MEMFRAC" \
    --max-running-requests "$CONC" --disable-cuda-graph --attention-backend flashinfer \
    > "$SRVLOG" 2>&1 &
  local SRV_PID=$!
  cleanup() { kill "$SRV_PID" 2>/dev/null; kill_servers; }
  trap cleanup EXIT
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[lc] died:"; tail -30 "$SRVLOG"; trap - EXIT; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[lc] TIMEOUT"; cleanup; trap - EXIT; return 1; fi
  done
  echo "[lc] ready after ${t}s"; sleep 3
  python "$FVL/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$CONC" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" >/dev/null 2>&1 || true
  python "$FVL/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$TOTAL" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" "$RESULT"
  cleanup; trap - EXIT
}

for c in $CONC_LIST; do run_one "$c"; done
echo "[lc] DONE"
