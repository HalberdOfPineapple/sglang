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
    dllm_step_counter,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
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

        # D2 step counter (no-op unless SGLANG_DLLM_PROFILE=1). n_committed = the
        # masked positions each block must decode = block_size - start. finish_steps
        # is filled in below as each block becomes mask-free; steps_executed counts
        # the forwards the shared loop runs before all blocks finish (= S_k).
        step_counter = dllm_step_counter(model_runner.tp_rank)
        if step_counter is not None:
            call_id = step_counter.next_call()
            n_committed = [self.block_size - s for s in start_list]
            finish_steps: List[Union[int, None]] = [None] * batch_size
            steps_executed = 0

        for step in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break
            if step_counter is not None:
                steps_executed = step + 1
                committed_step = 0  # tokens committed across the batch this step
                n_active_step = 0   # blocks that still had masks this step

            with dllm_nvtx_range(f"dllm_forward.step{step}"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)


                
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            dllm_nvtx_push(f"dllm_select.step{step}")
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    if step_counter is not None and finish_steps[batch_id] is None:
                        finish_steps[batch_id] = step  # already mask-free this step
                    continue
                if step_counter is not None:
                    n_active_step += 1
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
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

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

                if step_counter is not None:
                    committed_step += int(transfer_index.sum().item())
                    if finish_steps[batch_id] is None:
                        # commit happened in place above; block is done if no mask left
                        if torch.sum(block_input_ids == self.mask_id).item() == 0:
                            finish_steps[batch_id] = step
            dllm_nvtx_pop()
            if step_counter is not None:
                step_counter.log_step(
                    call_id, step, batch_size, n_active_step, committed_step
                )

        if step_counter is not None:
            step_counter.log_block(
                call_id, batch_size, self.block_size, steps_executed,
                finish_steps, n_committed
            )

        with dllm_nvtx_range("dllm_final_forward"):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
