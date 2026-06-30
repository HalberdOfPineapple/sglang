#!/usr/bin/env bash
# FOCUS vs LowConfidence — end-to-end throughput/latency comparison on the
# paper-exact reduced forward (LLaDA2.0-mini, 1xA100 80GB, TP=1, eager).
#
# For each algorithm and each concurrency, launch a server, drive a sustained
# HumanEval workload (driver keeps `conc` requests in flight), and record tok/s +
# per-request latency. FOCUS additionally logs the per-reduced-step redundancy
# Sigma|S|/(B*bs) (SGLANG_FOCUS_LOG_REDUNDANCY=1) which explains the speedup.
#
# dLLM forces eager (--disable-cuda-graph) + --attention-backend flashinfer
# (see memory dllm-eager-needs-flashinfer-attn). FOCUS forces the paged FlashInfer
# path internally (reduced phases read K/V from cache).
#
# Scripts in repo (this dir); OUTPUTS (logs, result JSON = data) mirror to:
#   data : $DATA_ROOT/profiling/dllm/focus_vs_lowconf/{logs,profiles}/
#
# Usage:  ./run_focus_vs_lowconf.sh            # full sweep, both algos
#         CONC_LIST="1 8" ./run_focus_vs_lowconf.sh
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/focus_vs_lowconf}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
LOGS=${LOGS:-$OUT/logs}
HERE="$REPO/experiments/$EXP_PATH"
PORT=${PORT:-30030}
TP=${TP:-1}
MODEL=${MODEL:-/cephfs/shared/model/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC_LIST=${CONC_LIST:-"1 8 16"}
HUMANEVAL=${HUMANEVAL:-/cephfs/shared/wxli/human-eval/data/HumanEval.jsonl.gz}
N_SAMPLES=${N_SAMPLES:-20}
MAX_NEW=${MAX_NEW:-128}              # 4 blocks of 32 per request
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HOST=127.0.0.1

mkdir -p "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HOME=${HF_HOME:-/cephfs/shared/huggingface}

# Kill any stale SGLang server before starting (avoids cross-run GPU OOM).
pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 3

echo "[fvl] repo=$REPO data=$OUT model=$MODEL TP=$TP conc='$CONC_LIST' max_new=$MAX_NEW"
echo "[fvl] commit=$(cd "$REPO" && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Bulletproof teardown: kill every sglang server and WAIT until the GPU is free
# and the port is released, so the next run never races a leftover server (which
# silently serves requests with the wrong algorithm/env and corrupts results).
kill_servers() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "sglang::" 2>/dev/null
  local t=0
  while :; do
    local mem; mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local procs; procs=$(pgrep -fc "sglang.launch_server" || echo 0)
    if [ "${mem:-9999}" -lt 2000 ] && [ "${procs:-0}" -eq 0 ]; then break; fi
    sleep 3; t=$((t+3)); [ "$t" -ge 120 ] && break
  done
}

run_one() {  # $1=ALGO (Focus|LowConfidence)  $2=concurrency
  local ALGO=$1 CONC=$2
  local CFG TAG
  if [ "$ALGO" = "Focus" ]; then CFG="$HERE/configs/focus.yaml"; else CFG="$HERE/configs/low_confidence.yaml"; fi
  TAG="${ALGO,,}_c${CONC}"
  local SRVLOG="$LOGS/${TAG}_server.log"
  local DRIVELOG="$LOGS/${TAG}_drive.log"
  local RESULT="$LOGS/${TAG}_result.json"
  local TOTAL=$(( CONC * 3 )); [ "$TOTAL" -lt 6 ] && TOTAL=6
  echo "=============================================================="
  echo "[fvl] ALGO=$ALGO conc=$CONC total_reqs=$TOTAL -> $RESULT"

  local REDCSV="$LOGS/${TAG}_redundancy.csv"
  if [ "$ALGO" = "Focus" ]; then
    export SGLANG_FOCUS_LOG_REDUNDANCY=1 SGLANG_FOCUS_REDUNDANCY_CSV="$REDCSV"
    rm -f "$REDCSV"
  else
    unset SGLANG_FOCUS_LOG_REDUNDANCY SGLANG_FOCUS_REDUNDANCY_CSV
  fi

  # Guarantee a clean slate (no leftover server racing this run's port/GPU).
  kill_servers

  # PYTHONUNBUFFERED so the scheduler subprocess flushes [focus] prints promptly.
  PYTHONUNBUFFERED=1 python -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --dllm-algorithm "$ALGO" --dllm-algorithm-config "$CFG" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP" --mem-fraction-static "$MEMFRAC" \
    --max-running-requests "$CONC" \
    --disable-cuda-graph --attention-backend flashinfer \
    > "$SRVLOG" 2>&1 &
  local SRV_PID=$!
  cleanup() { kill "$SRV_PID" 2>/dev/null; kill_servers; }
  trap cleanup EXIT

  echo "[fvl] waiting for server (timeout ${READY_TIMEOUT}s)..."
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[fvl] server died:"; tail -40 "$SRVLOG"; trap - EXIT; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[fvl] TIMEOUT"; tail -40 "$SRVLOG"; cleanup; trap - EXIT; return 1; fi
  done
  echo "[fvl] server ready after ${t}s"
  sleep 3  # settle before driving

  # warmup (not measured) — fill caches at this batch size
  python "$HERE/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$CONC" \
    "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" >/dev/null 2>&1 || true

  echo "[fvl] measured workload (sustained conc=$CONC, $TOTAL reqs)..."
  python "$HERE/drive_humaneval.py" "$HOST" "$PORT" "$CONC" "$TOTAL" \
    "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" "$RESULT" 2>&1 | tee "$DRIVELOG"

  if [ "$ALGO" = "Focus" ]; then
    echo "[fvl] FOCUS redundancy CSV rows: $(( $(wc -l < "$REDCSV" 2>/dev/null || echo 1) - 1 ))"
  fi

  cleanup; trap - EXIT
}

for c in $CONC_LIST; do
  run_one LowConfidence "$c"
  run_one Focus "$c"
done

echo "[fvl] ALL DONE — parse:"
python "$HERE/parse_focus_vs_lowconf.py" "$LOGS" 2>&1 | tee "$LOGS/focus_vs_lowconf_summary.txt" || true
