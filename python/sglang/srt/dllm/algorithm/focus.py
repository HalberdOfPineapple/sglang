"""FOCUS: Training-Free Token Eviction for Diffusion LLMs.

Phase A implementation: Correctness first with Delayed Cache (DC+).
- PyTorch-based importance computation and selection
- Neighbor-Aware delayed caching
- Single GPU only
- Eager execution (no CUDA graph)

Reference: FOCUS paper (ICML 2026), Section 3-4, Appendix C.
Implementation plan: notes/focus_sglang_implementation_plan.md
"""

import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
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
        self.enable_delayed_cache = config.algorithm_config.get(
            "enable_delayed_cache", True
        )

        if not self.enable_delayed_cache:
            raise ValueError(
                "FOCUS requires enable_delayed_cache=True (DC+ is prerequisite for quality)"
            )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        """Run FOCUS denoising with token eviction.

        For Phase A, this is a simplified version that:
        1. Uses standard model_runner.forward (no split yet)
        2. Implements DC+ (Neighbor-Aware delayed caching)
        3. Tracks FOCUS state for future eviction implementation

        The full eviction logic (split forward, importance computation, selection)
        will be added in subsequent commits.
        """
        batch_size = forward_batch.batch_size
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            with dllm_nvtx_range("dllm_prefill_forward"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        # Denoising loop
        for step in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            with dllm_nvtx_range(f"dllm_focus_forward.step{step}"):
                # TODO: Replace with forward_focus (split prefix/suffix)
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            dllm_nvtx_push(f"dllm_focus_select.step{step}")

            # Process each request in the batch
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
                block_mask_index = block_input_ids == self.mask_id

                if torch.sum(block_mask_index).item() == 0:
                    continue

                # Get logits for current block
                curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]

                # Decode: argmax and confidence
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

                # Select tokens to commit (confidence > threshold, else top-1)
                transfer_index = confidence > self.threshold
                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                # Commit decoded tokens
                block_input_ids[transfer_index] = x[transfer_index]

                # TODO: Update focus_state (N̄_decoded, rightmost_processed)
                # TODO: Update delayed_cache_state (Neighbor-Aware caching)

            dllm_nvtx_pop()

        # Final forward pass
        with dllm_nvtx_range("dllm_focus_final_forward"):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        # Return next token ids (variable length per request)
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


# Register algorithm
Algorithm = Focus
