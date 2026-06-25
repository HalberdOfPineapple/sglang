# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lightweight, env-gated NVTX instrumentation for the dLLM denoising loop.

These helpers annotate the *algorithm-level* control flow (the per-denoising-step
forward / host-selection phases in ``dllm/algorithm/*``) with NVTX ranges so that
an ``nsys`` capture can attribute GPU kernels — in particular NCCL collectives
(TP all-reduce, EP all-to-all) — to individual denoising iterations and separate
exposed communication from compute.

Design goals:
- **Zero overhead when off.** Everything is a no-op unless ``SGLANG_DLLM_NVTX=1``.
- **No model-module intrusion.** Ranges are emitted from the dLLM algorithm code
  only; NCCL collectives are tagged automatically by nsys (kernel names), so the
  model modules (``models/llada2.py``) need no annotation.

Why a dedicated flag instead of ``--enable-layerwise-nvtx-marker`` (which wraps
*every* nn.Module via ``utils/nvtx_pytorch_hooks.PytHooks``): the layerwise marker
is far too fine-grained for a step-level view and adds push/pop on every module,
obscuring the denoising-loop structure we care about for experiment D1.
"""

import csv
import os
from contextlib import contextmanager
from typing import List, Optional

import torch.cuda.nvtx as nvtx

# Read once at import. Set SGLANG_DLLM_NVTX=1 in the scheduler/TP-worker process
# environment to enable; leave unset for bit-identical baseline behaviour.
DLLM_NVTX_ENABLED: bool = os.environ.get("SGLANG_DLLM_NVTX", "0") == "1"

# Step/S_k counter (experiment D2). Set SGLANG_DLLM_PROFILE=1 to log, per block,
# how many denoising steps (``S_k``) the LowConfidence loop spends and when each
# request's block becomes mask-free. This is the ONLY quantity D2 needs that is
# not in ``meta_info`` (the comm_per_step constant comes from the D1 nsys trace).
# Like the NVTX ranges above it is a no-op unless the flag is set, so the baseline
# path stays bit-identical (CLAUDE.md isolation).
DLLM_PROFILE_ENABLED: bool = os.environ.get("SGLANG_DLLM_PROFILE", "0") == "1"


@contextmanager
def dllm_nvtx_range(name: str):
    """Context manager emitting an NVTX range ``name`` when enabled, else a no-op."""
    if DLLM_NVTX_ENABLED:
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def dllm_nvtx_push(name: str) -> None:
    """Open an NVTX range. Must be paired with :func:`dllm_nvtx_pop`."""
    if DLLM_NVTX_ENABLED:
        nvtx.range_push(name)


def dllm_nvtx_pop() -> None:
    """Close the most recent :func:`dllm_nvtx_push` range."""
    if DLLM_NVTX_ENABLED:
        nvtx.range_pop()


class DllmStepCounter:
    """Append-only CSV logger for the dLLM denoising loop (experiment D2).

    Writes TWO CSVs so per-token comm/comp can be formed PER STEP (tying each
    forward to the tokens it actually decoded) rather than block-level S_k times a
    single global average — the D2 methodology fix.

    Per-step CSV (``*_perstep.csv``) — one row per denoising forward:
      - ``call_id``   : the run() invocation this step belongs to
      - ``step``      : 0-based step index within the call
      - ``batch_size``: blocks forwarded this step (= the running batch; the FULL
                        block is forwarded every step, so this sets the forward's
                        comm/comp cost — it maps to the captured CUDA graph)
      - ``n_active``  : blocks that still had masks this step (<= batch_size)
      - ``committed`` : tokens committed across the batch THIS step (the per-step
                        denominator for comm/comp-per-token; varies 1..many)

    Per-block CSV (``*`` as given) — one row per (call, request-block):
      - ``S_k``       : denoising steps the *batch* ran before exit (straggler-incl.)
      - ``finish_step``: step at which THIS block became mask-free (intrinsic s_k-1)
      - ``n_committed``: masked positions this block decoded (tokens delivered)
    Only rank 0 writes (all TP ranks run the identical loop). Flushed per call.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.call_id = -1
        step_path = (path[:-4] if path.endswith(".csv") else path) + "_perstep.csv"
        self._bf = open(path, "w", newline="")
        self._bw = csv.writer(self._bf)
        self._bw.writerow(
            ["call_id", "batch_id", "batch_size", "block_size",
             "S_k", "finish_step", "n_committed"]
        )
        self._bf.flush()
        self._sf = open(step_path, "w", newline="")
        self._sw = csv.writer(self._sf)
        self._sw.writerow(["call_id", "step", "batch_size", "n_active", "committed"])
        self._sf.flush()

    def next_call(self) -> int:
        self.call_id += 1
        return self.call_id

    def log_step(self, call_id: int, step: int, batch_size: int,
                 n_active: int, committed: int) -> None:
        self._sw.writerow([call_id, step, batch_size, n_active, committed])

    def log_block(
        self,
        call_id: int,
        batch_size: int,
        block_size: int,
        steps_executed: int,
        finish_steps: List[Optional[int]],
        n_committed: List[int],
    ) -> None:
        for b in range(batch_size):
            fs = finish_steps[b]
            self._bw.writerow(
                [call_id, b, batch_size, block_size, steps_executed,
                 -1 if fs is None else fs, n_committed[b]]
            )
        self._bf.flush()
        self._sf.flush()


_STEP_COUNTER: Optional[DllmStepCounter] = None


def dllm_step_counter(tp_rank: int) -> Optional[DllmStepCounter]:
    """Return the rank-0 step counter (lazily created), or ``None`` when disabled.

    Path comes from ``SGLANG_DLLM_PROFILE_CSV`` (default ``dllm_steps.csv`` in CWD);
    the run script points it at the D2 data dir on CephFS. Non-zero ranks always get
    ``None`` so only one process writes the (rank-symmetric) schedule.
    """
    if not DLLM_PROFILE_ENABLED or tp_rank != 0:
        return None
    global _STEP_COUNTER
    if _STEP_COUNTER is None:
        path = os.environ.get("SGLANG_DLLM_PROFILE_CSV", "dllm_steps.csv")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _STEP_COUNTER = DllmStepCounter(path)
    return _STEP_COUNTER
