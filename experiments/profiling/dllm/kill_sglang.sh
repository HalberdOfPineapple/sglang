#!/usr/bin/env bash
# Kill any running SGLang server (launcher + TP/EP worker subprocesses) and the nsys
# profiler wrapper our D1/D2 runs launch it under. Graceful first (SIGTERM/SIGINT so
# nsys can finalize its .nsys-rep), then SIGKILL the survivors after a short wait.
#
# Usage:
#   bash experiments/profiling/dllm/kill_sglang.sh           # graceful then force
#   bash experiments/profiling/dllm/kill_sglang.sh -9        # force kill immediately
#   GRACE=15 bash .../kill_sglang.sh                         # wait 15s before SIGKILL
set -uo pipefail

GRACE=${GRACE:-8}
FORCE=0
[ "${1:-}" = "-9" ] && FORCE=1

# SGLang worker proctitles (sglang::scheduler_TP0, sglang::detokenizer, ...) + the
# launcher/bench entrypoints + our nsys wrapper. Exclude THIS script and the pgrep.
SGL_RE='sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::'
NSYS_RE='nsys.*(profile|launch).*sglang|[ /]nsys .*launch_server'
SELF=$$

pids() {  # all matching pids, minus this script
  { pgrep -f "$SGL_RE"; pgrep -f "$NSYS_RE"; } 2>/dev/null | sort -un | grep -vw "$SELF"
}

mapfile -t P < <(pids)
if [ "${#P[@]}" -eq 0 ]; then
  echo "[kill_sglang] no SGLang/nsys processes running."
  exit 0
fi
echo "[kill_sglang] targets:"
ps -o pid,ppid,etime,cmd -p "$(IFS=,; echo "${P[*]}")" 2>/dev/null | sed 's/\(.\{120\}\).*/\1/'

if [ "$FORCE" -eq 0 ]; then
  echo "[kill_sglang] SIGINT/SIGTERM (grace ${GRACE}s; lets nsys flush its report)..."
  kill -INT "${P[@]}" 2>/dev/null
  kill -TERM "${P[@]}" 2>/dev/null
  for _ in $(seq 1 "$GRACE"); do
    sleep 1
    [ -z "$(pids)" ] && { echo "[kill_sglang] all exited cleanly."; break; }
  done
fi

mapfile -t LEFT < <(pids)
if [ "${#LEFT[@]}" -gt 0 ]; then
  echo "[kill_sglang] SIGKILL survivors: ${LEFT[*]}"
  kill -9 "${LEFT[@]}" 2>/dev/null
  sleep 1
fi

REM="$(pids)"
[ -z "$REM" ] && echo "[kill_sglang] done — none remaining." || echo "[kill_sglang] WARNING still alive: $REM"
echo "[kill_sglang] GPU memory now:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
