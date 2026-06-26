#!/usr/bin/env bash
# D2 — S_k x amplification: exposed comm/comp per OUTPUT TOKEN, measured PER STEP.
# (see experiments/profiling/dllm/dllm_baseline_profiling_plan.md, D2.)
#
# Report: experiments/profiling/dllm/d2_sk_amplification/h100/README.md
#
# Method (rigorous redo): for every denoising forward we record, with the §3
# counter (SGLANG_DLLM_PROFILE=1), the batch_size and the tokens committed THAT
# step (per-step CSV), and we CO-RUN nsys (--cuda-graph-trace=node, like D1) to
# measure comm/comp time of the forward AT THIS RUN's batch size. Joining the two
# gives comm-per-token / comp-per-token as a per-step DISTRIBUTION (not block-S_k
# times one global average). Workload: HumanEval prompts driven at SUSTAINED
# concurrency so the batch stays full (GPU-bound). Swept over concurrency 4/8/16.
#
# Scripts in repo (this dir); OUTPUTS (profiles, CSVs, logs = data) mirror to:
#   data : $DATA_ROOT/profiling/dllm/d2_sk_amplification/h100/{profiles,logs}/
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/d2_sk_amplification/h100}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
PROF=${PROF:-$OUT/profiles}
LOGS=${LOGS:-$OUT/logs}
PORT=${PORT:-30000}
TP=${TP:-4}; EP=${EP:-4}
MODEL=${MODEL:-inclusionAI/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC_LIST=${CONC_LIST:-"4 8 16"}     # concurrency sweep (= max-running-requests each)
HUMANEVAL=${HUMANEVAL:-/cephfs/shared/wxli/human-eval/data/HumanEval.jsonl.gz}
N_SAMPLES=${N_SAMPLES:-20}           # HumanEval prompts to cycle
MAX_NEW=${MAX_NEW:-256}              # 8 blocks of 32 per request
READY_TIMEOUT=${READY_TIMEOUT:-2400}
HOST=127.0.0.1

mkdir -p "$PROF" "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export SGLANG_DLLM_NVTX=1            # per-step NVTX ranges (for nsys per-phase)
export SGLANG_DLLM_PROFILE=1         # the §3 per-step + per-block counter
export SGLANG_TORCH_PROFILER_DIR=$PROF

run_one() {  # $1 = concurrency
  local CONC=$1
  local TAG=d2_h100_tp4_c${CONC}
  local REP=$PROF/${TAG}
  local SRVLOG=$LOGS/${TAG}_server.log
  local TOTAL=$(( CONC * 4 ))        # measured requests (>= 2x conc -> steady full batch)
  export SGLANG_DLLM_PROFILE_CSV=$LOGS/${TAG}_blocks.csv   # perstep CSV derived from this
  echo "=============================================================="
  echo "[d2] TAG=$TAG conc=$CONC maxreq=$CONC total_reqs=$TOTAL max_new=$MAX_NEW"
  echo "[d2] report -> ${REP}.nsys-rep ; csv -> ${SGLANG_DLLM_PROFILE_CSV%.csv}{,_perstep}.csv"

  nsys profile -t cuda,nvtx,nccl --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    --force-overwrite true -o "$REP" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --dllm-algorithm LowConfidence \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code \
      --tp-size "$TP" --ep-size "$EP" \
      --mem-fraction-static "$MEMFRAC" --max-running-requests "$CONC" \
      > "$SRVLOG" 2>&1 &
  NSYS_PID=$!
  cleanup() { kill "${NSYS_PID:-}" 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; }
  trap cleanup EXIT

  echo "[d2] waiting for server (timeout ${READY_TIMEOUT}s)..."
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$NSYS_PID" 2>/dev/null; then echo "[d2] server died:"; tail -40 "$SRVLOG"; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[d2] TIMEOUT"; tail -40 "$SRVLOG"; return 1; fi
  done
  echo "[d2] server ready after ${t}s"

  # warmup (NOT captured) — fill caches/graphs at this batch size
  python "$REPO/experiments/$EXP_PATH/drive_humaneval.py" \
    "$HOST" "$PORT" "$CONC" "$CONC" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" || true

  echo "[d2] /start_profile -> nsys capture begins"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/start_profile" \
    -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}' >/dev/null

  echo "[d2] measured workload (sustained conc=$CONC, $TOTAL reqs)..."
  python "$REPO/experiments/$EXP_PATH/drive_humaneval.py" \
    "$HOST" "$PORT" "$CONC" "$TOTAL" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" \
    2>&1 | tee -a "$LOGS/${TAG}_drive.log"

  echo "[d2] /stop_profile -> nsys stops + shuts server down"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/stop_profile" >/dev/null 2>&1 || true
  wait "$NSYS_PID" 2>/dev/null
  trap - EXIT
  if [ ! -f "${REP}.nsys-rep" ]; then echo "[d2] ERROR: ${REP}.nsys-rep missing"; tail -40 "$SRVLOG"; return 1; fi

  echo "[d2] nsys stats..."
  nsys stats --force-export=true \
    --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum --format csv \
    --output "$REP" "${REP}.nsys-rep" > "$LOGS/${TAG}_stats.log" 2>&1

  echo "[d2] parse..."
  python "$REPO/experiments/$EXP_PATH/parse_d2.py" \
    "${SGLANG_DLLM_PROFILE_CSV%.csv}" "$REP" 2>&1 | tee "$LOGS/${TAG}_summary.txt" || true
  pkill -f "sglang.launch_server" 2>/dev/null; sleep 3
}

for c in $CONC_LIST; do run_one "$c"; done
echo "[d2] ALL DONE — cross-concurrency table:"
python "$REPO/experiments/$EXP_PATH/parse_d2.py" --sweep "$LOGS" 2>&1 | tee "$LOGS/d2_sweep_summary.txt" || true
