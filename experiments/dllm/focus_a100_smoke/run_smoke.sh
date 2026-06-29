#!/usr/bin/env bash
# FOCUS Phase-A single-GPU smoke test (1x A100 80GB, LLaDA2.0-mini, TP=1).
#
# Launches an SGLang server with a chosen dLLM algorithm/config, sends a few
# fixed prompts, and prints the generations. Use it to (a) confirm FOCUS runs
# end-to-end, and (b) confirm FOCUS @ alpha->inf matches LowConfidence.
#
# Usage:
#   ./run_smoke.sh focus                # FOCUS with eviction (alpha=1.5)
#   ./run_smoke.sh focus_alpha_inf      # FOCUS, no eviction (== LowConfidence)
#   ./run_smoke.sh low_confidence       # baseline
#
# Outputs (logs only; small + reproducible -> kept in-repo under logs/):
#   logs/<algo>_server.log , logs/<algo>_gen.json
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
HERE="$REPO/experiments/dllm/focus_a100_smoke"
MODEL=${MODEL:-/cephfs/shared/model/LLaDA2.0-mini}
PORT=${PORT:-31000}
MEMFRAC=${MEMFRAC:-0.7}            # sparse-MoE OOM headroom on a single card
HOST=127.0.0.1
READY_TIMEOUT=${READY_TIMEOUT:-1200}

WHICH=${1:-focus}
case "$WHICH" in
  focus)            ALGO=Focus;          CFG="$HERE/configs/focus.yaml" ;;
  focus_alpha_inf)  ALGO=Focus;          CFG="$HERE/configs/focus_alpha_inf.yaml" ;;
  low_confidence)   ALGO=LowConfidence;  CFG="$HERE/configs/low_confidence.yaml" ;;
  *) echo "unknown variant: $WHICH"; exit 2 ;;
esac

mkdir -p "$HERE/logs"
SRVLOG="$HERE/logs/${WHICH}_server.log"
GENOUT="$HERE/logs/${WHICH}_gen.json"

export HF_HUB_DISABLE_XET=1
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HOME=${HF_HOME:-/cephfs/shared/huggingface}

echo "[smoke] variant=$WHICH algo=$ALGO cfg=$CFG model=$MODEL"
echo "[smoke] launching server -> $SRVLOG"

python -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --dllm-algorithm "$ALGO" --dllm-algorithm-config "$CFG" \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests 4 \
  --disable-cuda-graph --attention-backend flashinfer \
  > "$SRVLOG" 2>&1 &
SRV_PID=$!

cleanup() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[smoke] waiting for readiness (timeout ${READY_TIMEOUT}s)..."
deadline=$(( SECONDS + READY_TIMEOUT ))
until curl -s --noproxy '*' "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[smoke] SERVER DIED -- tail of log:"; tail -40 "$SRVLOG"; exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "[smoke] TIMEOUT -- tail of log:"; tail -40 "$SRVLOG"; exit 1
  fi
  sleep 3
done
echo "[smoke] server ready."

python "$HERE/drive_smoke.py" --host "$HOST" --port "$PORT" --out "$GENOUT"
RC=$?
echo "[smoke] driver rc=$RC ; generations -> $GENOUT"
exit $RC
