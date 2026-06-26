#!/usr/bin/env bash
# D2 / A100-PCIe leaf — comm VOLUME + PCIe comm TIME, with an NVLink-time projection.
# (see experiments/profiling/dllm/dllm_baseline_profiling_plan.md, D2; and the
#  H100/NVLink sibling experiments/profiling/dllm/d2_sk_amplification/h100/README.md)
#
# Report: experiments/profiling/dllm/d2_sk_amplification/a100/README.md
#
# Extension over the H100 D2 run: (1) re-run the SAME per-step measurement on
# 4xA100 80GB PCIe (no NVLink) and add a batch_size=1 point; (2) MEASURE the real
# A100 compute time per forward (interconnect-independent, per CUDA-graph replay)
# and the analytic comm VOLUME (deterministic: 41 all-reduces x bs*block*hidden*2B
# + 1 vocab all-gather x bs*block*vocab*2B); (3) PROJECT the comm time onto A100
# NVLink (projected_comm = bus_traffic / busbw) and form the D2-style comm fraction
# comm/(comm+compute) against the MEASURED A100 compute -- the comm/compute balance
# this box WOULD have on NVLink. The raw PCIe comm time is interconnect-bound and
# DISCARDED (comparing PCIe-vs-NVLink is meaningless). Data under the a100/ mirror.
#
# Scripts in repo (this dir); OUTPUTS (profiles, CSVs, logs = data) mirror to:
#   data : $DATA_ROOT/profiling/dllm/d2_sk_amplification/a100/{profiles,logs}/
set -uo pipefail

REPO=${REPO:-/root/sglang_a100/sglang}
DATA_ROOT=${DATA_ROOT:-/cephfs/shared/wxli/sglang-dllm}
EXP_PATH=${EXP_PATH:-profiling/dllm/d2_sk_amplification/a100}
OUT=${OUT:-$DATA_ROOT/$EXP_PATH}
PROF=${PROF:-$OUT/profiles}
LOGS=${LOGS:-$OUT/logs}
PORT=${PORT:-30000}
TP=${TP:-4}; EP=${EP:-4}
MODEL=${MODEL:-inclusionAI/LLaDA2.0-mini}
MEMFRAC=${MEMFRAC:-0.7}
CONC_LIST=${CONC_LIST:-"1 4 8 16"}   # batch-size sweep (= max-running-requests each)
HUMANEVAL=${HUMANEVAL:-/cephfs/shared/wxli/human-eval/data/HumanEval.jsonl.gz}
N_SAMPLES=${N_SAMPLES:-20}           # HumanEval prompts to cycle
MAX_NEW=${MAX_NEW:-256}              # 8 blocks of 32 per request
READY_TIMEOUT=${READY_TIMEOUT:-2400}
HOST=127.0.0.1
# nsys is not on PATH in this container's batch shell; fall back to the copy
# bundled with the eval_310 env's nsight-compute (verified: 2025.1.1, supports
# --cuda-graph-trace=node + --capture-range=cudaProfilerApi).
NSYS=${NSYS:-$(command -v nsys || echo /root/miniconda3/envs/eval_310/nsight-compute-2025.1.1/host/target-linux-x64/nsys)}

mkdir -p "$PROF" "$LOGS"
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export SGLANG_DLLM_NVTX=1            # per-step NVTX ranges (for nsys per-phase)
export SGLANG_DLLM_PROFILE=1         # the §3 per-step + per-block counter
export SGLANG_TORCH_PROFILER_DIR=$PROF

echo "[a100] nsys = $NSYS"; "$NSYS" --version | head -1
echo "[a100] GPU/topology (interconnect = PCIe, NOT NVLink):"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -6

run_one() {  # $1 = concurrency / batch size
  local CONC=$1
  local TAG=d2_a100_tp4_c${CONC}
  local REP=$PROF/${TAG}
  local SRVLOG=$LOGS/${TAG}_server.log
  local TOTAL=$(( CONC * 4 )); [ "$TOTAL" -lt 8 ] && TOTAL=8   # >= steady full batch / enough fwds
  export SGLANG_DLLM_PROFILE_CSV=$LOGS/${TAG}_blocks.csv       # perstep CSV derived from this
  echo "=============================================================="
  echo "[a100] TAG=$TAG bs=$CONC maxreq=$CONC total_reqs=$TOTAL max_new=$MAX_NEW"
  echo "[a100] report -> ${REP}.nsys-rep ; csv -> ${SGLANG_DLLM_PROFILE_CSV%.csv}{,_perstep}.csv"

  # NOTE: this nsys build (2025.1.1, bundled with nsight-compute) has no 'nccl'
  # trace plugin. We don't need it: NCCL collectives run as CUDA device kernels
  # (ncclDevKernel_AllReduce_*/AllGather_*) captured by the 'cuda' trace, and the
  # parser classifies comm vs comp by kernel name from CUPTI_ACTIVITY_KIND_KERNEL.
  "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    --force-overwrite true -o "$REP" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --dllm-algorithm LowConfidence \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code \
      --tp-size "$TP" --ep-size "$EP" --attention-backend flashinfer \
      --mem-fraction-static "$MEMFRAC" --max-running-requests "$CONC" \
      > "$SRVLOG" 2>&1 &
  NSYS_PID=$!
  cleanup() { kill "${NSYS_PID:-}" 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; }
  trap cleanup EXIT

  echo "[a100] waiting for server (timeout ${READY_TIMEOUT}s)..."
  local t=0
  until curl -s --noproxy '*' "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$NSYS_PID" 2>/dev/null; then echo "[a100] server died:"; tail -40 "$SRVLOG"; return 1; fi
    if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "[a100] TIMEOUT"; tail -40 "$SRVLOG"; return 1; fi
  done
  echo "[a100] server ready after ${t}s"

  # warmup (NOT captured) — fill caches/graphs at this batch size
  python "$REPO/experiments/$EXP_PATH/drive_humaneval.py" \
    "$HOST" "$PORT" "$CONC" "$CONC" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" || true

  echo "[a100] /start_profile -> nsys capture begins"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/start_profile" \
    -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}' >/dev/null

  echo "[a100] measured workload (sustained bs=$CONC, $TOTAL reqs)..."
  python "$REPO/experiments/$EXP_PATH/drive_humaneval.py" \
    "$HOST" "$PORT" "$CONC" "$TOTAL" "$HUMANEVAL" "$N_SAMPLES" "$MAX_NEW" \
    2>&1 | tee -a "$LOGS/${TAG}_drive.log"

  echo "[a100] /stop_profile -> nsys stops + shuts server down"
  curl -s --noproxy '*' -X POST "http://$HOST:$PORT/stop_profile" >/dev/null 2>&1 || true
  wait "$NSYS_PID" 2>/dev/null
  trap - EXIT
  if [ ! -f "${REP}.nsys-rep" ]; then echo "[a100] ERROR: ${REP}.nsys-rep missing"; tail -40 "$SRVLOG"; return 1; fi

  echo "[a100] nsys stats..."
  "$NSYS" stats --force-export=true \
    --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum --format csv \
    --output "$REP" "${REP}.nsys-rep" > "$LOGS/${TAG}_stats.log" 2>&1

  echo "[a100] parse..."
  python "$REPO/experiments/$EXP_PATH/parse_a100.py" \
    "${SGLANG_DLLM_PROFILE_CSV%.csv}" "$REP" 2>&1 | tee "$LOGS/${TAG}_summary.txt" || true
  pkill -f "sglang.launch_server" 2>/dev/null; sleep 3
}

for c in $CONC_LIST; do run_one "$c"; done
echo "[a100] ALL DONE — cross-batch-size table:"
python "$REPO/experiments/$EXP_PATH/parse_a100.py" --sweep "$LOGS" 2>&1 | tee "$LOGS/d2_a100_sweep_summary.txt" || true
