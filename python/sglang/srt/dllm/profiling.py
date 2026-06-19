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

import os
from contextlib import contextmanager

import torch.cuda.nvtx as nvtx

# Read once at import. Set SGLANG_DLLM_NVTX=1 in the scheduler/TP-worker process
# environment to enable; leave unset for bit-identical baseline behaviour.
DLLM_NVTX_ENABLED: bool = os.environ.get("SGLANG_DLLM_NVTX", "0") == "1"


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
