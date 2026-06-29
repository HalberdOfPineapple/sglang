"""FOCUS: Training-Free Token Eviction for Diffusion LLMs.

Phase A implementation: Correctness first with Delayed Cache (DC+).
- PyTorch-based importance computation and selection
- Neighbor-Aware delayed caching state tracking
- Single GPU only
- Eager execution (no CUDA graph)

Phase A uses a *logit-masking* realization of eviction rather than a reduced
forward: every denoising step runs a full forward (with the importance
side-channel collecting Layer-0/Layer-1 intra-block scores), then FOCUS
selection decides which masked positions are *eligible to commit* this step.
Non-retained masked positions have their commit suppressed and are revisited
next step. This is functionally equivalent to the paper's eviction for the
purpose of *which tokens decode when* (and is bit-for-bit identical to
LowConfidence when alpha -> inf so that K = B), and isolates the selection /
budgeting / DC+ logic for validation before the performance-sensitive reduced
forward (Phase C) is wired in. See notes/focus_sglang_implementation_plan.md R1.

Reference: FOCUS paper (ICML 2026), Section 3-4, Appendix C.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.focus_utils import (
    FocusRuntimeView,
    compute_focus_targets,
    compute_should_evict,
    select_and_enforce_constraints,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.profiling import (
    dllm_nvtx_pop,
    dllm_nvtx_push,
    dllm_nvtx_range,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Focus(DllmAlgorithm):
    """FOCUS algorithm with token eviction and Neighbor-Aware delayed caching."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.alpha = config.algorithm_config.get("alpha", 1.5)
        self.maxpool_k = config.algorithm_config.get("maxpool_k", 3)
        self.min_retain = config.algorithm_config.get("min_retain", 1)
        self.importance_layers = tuple(
            config.algorithm_config.get("importance_layers", (0, 1))
        )
        self.enable_delayed_cache = config.algorithm_config.get(
            "enable_delayed_cache", True
        )
        # n_bar default at step 1 of each block (Eq. 4 says default 1).
        self.n_bar_init = config.algorithm_config.get("n_bar_init", 1.0)

    def _build_focus_view(
        self, batch_size: int, device: torch.device, avg_decoded: torch.Tensor
    ) -> FocusRuntimeView:
        """Build a FocusRuntimeView for a full-block forward (uniform CSR layout)."""
        seq_offsets = torch.arange(
            0, (batch_size + 1) * self.block_size, self.block_size,
            dtype=torch.int64, device=device,
        )
        return FocusRuntimeView(
            block_size=self.block_size,
            batch_size=batch_size,
            seq_offsets=seq_offsets,
            maxpool_k=self.maxpool_k,
            importance_layers=self.importance_layers,
            avg_decoded=avg_decoded,
        )

    def _select_retained(
        self,
        focus_view: FocusRuntimeView,
        forward_batch: ForwardBatch,
    ) -> List[torch.Tensor]:
        """Run FOCUS budgeting + selection. Returns per-request retained block indices.

        Importance was collected during the just-finished forward; here we form
        ΔI, compute the per-request budget (no N_σ), the should-evict flag, and
        the structurally-enforced retained set S (top-k OR N_σ expansion,
        AR-context, placeholder integrity). Mirrors the official kernels.
        """
        delta_I = focus_view.get_delta_importance()  # [batch * block_size]

        # mask[j] True iff position j is still a [MASK] (candidate to decode).
        mask = forward_batch.input_ids == self.mask_id

        # Per-request masked counts via CSR boundaries.
        seq_offsets = focus_view.seq_offsets
        batch_size = len(seq_offsets) - 1
        mask_lengths = torch.tensor(
            [
                int(mask[int(seq_offsets[b]):int(seq_offsets[b + 1])].sum().item())
                for b in range(batch_size)
            ],
            dtype=torch.int32,
            device=delta_I.device,
        )

        targets = compute_focus_targets(mask_lengths, focus_view.avg_decoded, self.alpha)
        should_evict = compute_should_evict(mask_lengths, targets)

        _, retained_maps = select_and_enforce_constraints(
            delta_I,
            mask,
            seq_offsets,
            targets,
            should_evict,
            self.block_size,
        )
        return retained_maps

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: no mask token -> single forward to populate KV cache (prefill).
        if torch.sum(mask_index).item() == 0:
            with dllm_nvtx_range("dllm_prefill_forward"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return logits_output, [], can_run_cuda_graph

        # start = #already-committed positions per block (prompt prefix inside block).
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        # Per-request cumulative decode statistics (Eq. 17-19). Tracked here for
        # Phase A; will be migrated onto Req.focus_state once the scheduler
        # feedback loop is wired in. token_sum/total_steps persist within this
        # run; n_bar defaults to n_bar_init on the first step.
        token_sum = [0] * batch_size
        total_steps = [0] * batch_size

        for step in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # n_bar per request: cumulative mean so far, or init at first step.
            avg_decoded = torch.tensor(
                [
                    (token_sum[b] / total_steps[b]) if total_steps[b] > 0
                    else self.n_bar_init
                    for b in range(batch_size)
                ],
                dtype=torch.float32,
                device=device,
            )

            focus_view = self._build_focus_view(batch_size, device, avg_decoded)
            forward_batch.focus_view = focus_view

            with dllm_nvtx_range(f"dllm_focus_forward.step{step}"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            forward_batch.focus_view = None

            dllm_nvtx_push(f"dllm_focus_select.step{step}")

            # FOCUS selection: which masked positions may commit this step.
            if focus_view.has_importance():
                retained_maps = self._select_retained(focus_view, forward_batch)
            else:
                # Importance side-channel did not fire (e.g. model without the
                # hook). Fall back to retaining all positions = LowConfidence.
                retained_maps = [None] * batch_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                # FOCUS eviction (logit-masking realization): only retained
                # masked positions are eligible to commit this step.
                retained = retained_maps[batch_id]
                if retained is not None:
                    eligible = torch.zeros_like(block_mask_index)
                    eligible[retained] = True
                    confidence = torch.where(
                        eligible, confidence, torch.full_like(confidence, -np.inf)
                    )

                transfer_index = confidence > self.threshold
                if transfer_index.sum().item() == 0:
                    # Guarantee progress: commit the single most-confident eligible
                    # (or, if none eligible, most-confident masked) position.
                    if torch.isinf(confidence).all():
                        confidence = torch.where(
                            block_mask_index, p, torch.full_like(p, -np.inf)
                        )
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                num_committed = int(transfer_index.sum().item())
                block_input_ids[transfer_index] = x[transfer_index]

                # Update per-request decode statistics for next step's budget.
                token_sum[batch_id] += num_committed
                total_steps[batch_id] += 1

            dllm_nvtx_pop()

        with dllm_nvtx_range("dllm_focus_final_forward"):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]
        return logits_output, next_token_ids_list, can_run_cuda_graph


# Register algorithm
Algorithm = Focus
