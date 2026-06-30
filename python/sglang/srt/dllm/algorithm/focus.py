"""FOCUS: Training-Free Token Eviction for Diffusion LLMs (paper-exact forward).

Each denoising step runs a *reduced* forward that physically evicts non-decodable
tokens after Layer 1, so Layers 1-attn..L execute on |S| ≪ B tokens (the source
of the FLOPs win — not the earlier logit-masking realization, which ran every
layer on all B tokens and saved nothing). The split (plan §8), per step:

  Phase P  (prefix) : embed → L0 full (collect I0, write L0 KV) → L1 QKV+RoPE
                      (collect I1) + write FULL-block L1 KV. q=B, kv=context+B.
  host             : ΔI=I1−I0 → targets/should_evict → select retained S →
                      compact q1/residual/positions/input_ids to |S|.
  Phase A1 (L1 attn): attention on |S| queries vs the full-block L1 KV (read-only)
                      + L1 MLP. q=|S|, kv=context+B.
  Phase S  (L2..L)  : ordinary layers on |S|; each writes its KV to the block's
                      contiguous prefix and reads context+|S|. norm + lm_head.

Anchor: α→∞ ⇒ targets=B ⇒ should_evict False ⇒ |S|=B ⇒ all three phases collapse
to the full forward ⇒ generations identical to LowConfidence.

The reduced phases read K/V from the cache (paged FlashInfer path), so FOCUS
forces ``attn_backend.use_paged = True`` (read dynamically per call).

Reference: FOCUS paper (ICML 2026), Section 3-4, Appendix C; official kernels in
~/FOCUS_ORIGIN/lmdeploy/pytorch/{kernels/cuda/focus.py,models/llada2.py}.
"""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.focus_forward import (
    PHASE_A1,
    PHASE_S,
    build_phase_s_out_cache_loc,
    make_focus_phase_batch,
)
from sglang.srt.dllm.algorithm.focus_reduce import build_retained_index
from sglang.srt.dllm.algorithm.focus_utils import (
    FocusRuntimeView,
    compute_focus_targets,
    compute_should_evict,
    select_and_enforce_constraints,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.profiling import dllm_nvtx_pop, dllm_nvtx_push, dllm_nvtx_range
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Focus(DllmAlgorithm):
    """FOCUS algorithm with paper-exact reduced (token-evicting) forward."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.alpha = config.algorithm_config.get("alpha", 1.5)
        self.maxpool_k = config.algorithm_config.get("maxpool_k", 3)
        self.min_retain = config.algorithm_config.get("min_retain", 1)
        self.importance_layers = tuple(
            config.algorithm_config.get("importance_layers", (0, 1))
        )
        # n_bar default at step 1 of each block (Eq. 4 says default 1).
        self.n_bar_init = config.algorithm_config.get("n_bar_init", 1.0)
        # SGLANG_FOCUS_LOG_REDUNDANCY=1 prints Σ|S|/(B·bs) per reduced step;
        # SGLANG_FOCUS_REDUNDANCY_CSV=<path> appends the same as CSV (robust to
        # scheduler-subprocess stdout capture). Both no-ops when unset.
        self._log_redundancy = os.environ.get("SGLANG_FOCUS_LOG_REDUNDANCY") == "1"
        self._redundancy_csv = os.environ.get("SGLANG_FOCUS_REDUNDANCY_CSV")
        self._redundancy_fh_cache = None

    # ------------------------------------------------------------------ helpers
    def _build_focus_view(
        self, batch_size: int, device: torch.device, avg_decoded: torch.Tensor
    ) -> FocusRuntimeView:
        """Uniform-CSR FocusRuntimeView for a full-block prefix forward."""
        seq_offsets = torch.arange(
            0,
            (batch_size + 1) * self.block_size,
            self.block_size,
            dtype=torch.int64,
            device=device,
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
        """
        ΔI → budget (no N_σ) → should_evict → enforced retained set S per req.
        Inputs:
            focus_view: CSR view of the full-block prefix forward.
            forward_batch: The forward batch containing input data.
        Outputs:
            retained_maps: List[LongTensor] sorted retained block indices.
        """
        delta_I = focus_view.get_delta_importance()  # [batch * block_size]
        mask = forward_batch.input_ids == self.mask_id
        seq_offsets = focus_view.seq_offsets
        batch_size = len(seq_offsets) - 1
        mask_lengths = torch.tensor(
            [
                int(mask[int(seq_offsets[b]) : int(seq_offsets[b + 1])].sum().item())
                for b in range(batch_size)
            ],
            dtype=torch.int32,
            device=delta_I.device,
        )
        targets = compute_focus_targets(
            mask_lengths, focus_view.avg_decoded, self.alpha
        )
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

    def _focus_reduced_forward(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        avg_decoded: torch.Tensor,
    ) -> Tuple[LogitsProcessorOutput, List[torch.Tensor], torch.Tensor]:
        """One reduced denoising forward. Returns (logits over |S|, retained_maps, new_lens).

        ``logits.full_logits`` is [Σ|S_b|, vocab], request-major, row j of request
        b corresponding to block position ``retained_maps[b][j]``.
        """
        model = model_runner.model
        attn_backend = model_runner.attn_backend
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # --- Phase P: full-block prefix (L0 full + L1 QKV+RoPE+fill). ---
        focus_view = self._build_focus_view(batch_size, device, avg_decoded)
        forward_batch.focus_view = focus_view
        attn_backend.init_forward_metadata(forward_batch)
        with dllm_nvtx_range("focus_prefix"):
            q1, residual = model.forward_focus_prefix(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
        forward_batch.focus_view = None

        # --- host: selection + compaction. ---
        retained_maps = self._select_retained(focus_view, forward_batch)
        keep_index, new_lens = build_retained_index(
            retained_maps, focus_view.seq_offsets
        )
        q1_s = q1.index_select(0, keep_index)
        residual_s = (
            None if residual is None else residual.index_select(0, keep_index)
        )
        positions_s = forward_batch.positions.index_select(0, keep_index)
        input_ids_s = forward_batch.input_ids.index_select(0, keep_index)

        # --- Phase A1: L1 attention on |S| vs full-block KV + L1 MLP. ---
        fb_a1 = make_focus_phase_batch(
            forward_batch, PHASE_A1, self.block_size, new_lens, input_ids_s, positions_s
        )
        attn_backend.init_forward_metadata(fb_a1)
        with dllm_nvtx_range("focus_l1_attn"):
            hidden_s, residual_s = model.forward_focus_l1_suffix(
                q1_s, fb_a1, residual_s
            )

        # --- Phase S: L2..L on |S| (KV → block prefix, read context+|S|). ---
        block_prefix_loc = build_phase_s_out_cache_loc(
            forward_batch.out_cache_loc, self.block_size, new_lens
        )
        fb_s = make_focus_phase_batch(
            forward_batch,
            PHASE_S,
            self.block_size,
            new_lens,
            input_ids_s,
            positions_s,
            block_prefix_loc,
        )
        attn_backend.init_forward_metadata(fb_s)
        with dllm_nvtx_range("focus_suffix"):
            logits_output = model.forward_focus_rest_and_logits(
                hidden_s, input_ids_s, positions_s, fb_s, residual_s
            )

        if self._log_redundancy or self._redundancy_csv:
            total = self.block_size * batch_size
            kept = int(new_lens.sum().item())
            ratio = kept / total
            if self._log_redundancy:
                print(
                    f"[focus] |S|/(B*bs) = {kept}/{total} = {ratio:.3f} "
                    f"(per-req |S|={new_lens.tolist()})",
                    flush=True,
                )
            if self._redundancy_csv:
                fh = self._redundancy_fh()
                if fh is not None:
                    fh.write(
                        f"{batch_size},{kept},{total},{ratio:.4f},"
                        f"\"{new_lens.tolist()}\"\n"
                    )
                    fh.flush()
        return logits_output, retained_maps, new_lens

    def _redundancy_fh(self):
        """Lazily open the redundancy CSV (header once); robust to subprocess stdout."""
        if self._redundancy_fh_cache is None and self._redundancy_csv:
            new = not os.path.exists(self._redundancy_csv)
            self._redundancy_fh_cache = open(self._redundancy_csv, "a")
            if new:
                self._redundancy_fh_cache.write("bs,kept,total,ratio,per_req_lens\n")
        return self._redundancy_fh_cache

    def _commit_step(
        self,
        forward_batch: ForwardBatch,
        full_logits_s: torch.Tensor,
        retained_maps: List[torch.Tensor],
        new_lens: torch.Tensor,
        token_sum: List[int],
        total_steps: List[int],
    ):
        """
        Decode the |S| retained logits and commit the confident masked ones.

        Only retained masked positions are eligible to decode this step (the
        evicted ones never produced suffix logits); each block commits all
        positions above ``threshold`` or, failing that, its single most-confident
        retained masked position (progress guarantee).
        """
        batch_size = forward_batch.batch_size
        offsets = torch.zeros(batch_size + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(new_lens.detach().cpu().to(torch.int64), dim=0)

        for b in range(batch_size):
            blk_start = b * self.block_size
            block_ids = forward_batch.input_ids[blk_start : blk_start + self.block_size]
            block_mask = block_ids == self.mask_id
            if int(block_mask.sum().item()) == 0:
                continue

            ret = retained_maps[b].to(forward_batch.input_ids.device)
            off = int(offsets[b].item())
            ret_logits = full_logits_s[off : off + int(new_lens[b].item())]
            if ret.numel() == 0:
                continue

            x_ret = torch.argmax(ret_logits, dim=-1)
            p_ret = torch.gather(
                F.softmax(ret_logits, dim=-1), -1, x_ret.unsqueeze(-1)
            ).squeeze(-1)

            # Scatter retained results back into [block_size]-shaped buffers.
            ret_is_masked = block_mask[ret]
            x_block = block_ids.clone()
            x_block[ret] = torch.where(ret_is_masked, x_ret, block_ids[ret])
            confidence = torch.full(
                (self.block_size,), -np.inf, device=block_ids.device
            )
            confidence[ret] = torch.where(
                ret_is_masked, p_ret, torch.full_like(p_ret, -np.inf)
            )

            transfer = confidence > self.threshold
            if int(transfer.sum().item()) == 0:
                if torch.isinf(confidence).all():
                    continue  # nothing retained is decodable this step
                _, top = torch.topk(confidence, k=1)
                transfer[top] = True

            num_committed = int(transfer.sum().item())
            block_ids[transfer] = x_block[transfer]
            token_sum[b] += num_committed
            total_steps[b] += 1

    # --------------------------------------------------------------------- run
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        # Reduced phases attend from the KV cache → force the paged FlashInfer path
        # (read dynamically each init_forward_metadata, so this takes effect now).
        model_runner.attn_backend.use_paged = True

        mask_index = forward_batch.input_ids == self.mask_id
        if torch.sum(mask_index).item() == 0:
            with dllm_nvtx_range("dllm_prefill_forward"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # start = #already-committed positions per block (prompt prefix inside block).
        start_list = []
        for block_id in range(batch_size):
            blk = forward_batch.input_ids[
                block_id * self.block_size : (block_id + 1) * self.block_size
            ]
            start_list.append(self.block_size - int((blk == self.mask_id).sum().item()))

        # Per-request cumulative decode statistics drive the budget K (Eq. 17-19).
        token_sum = [0] * batch_size
        total_steps = [0] * batch_size

        for step in range(self.block_size):
            if torch.sum(forward_batch.input_ids == self.mask_id).item() == 0:
                break

            avg_decoded = torch.tensor(
                [
                    (token_sum[b] / total_steps[b])
                    if total_steps[b] > 0
                    else self.n_bar_init
                    for b in range(batch_size)
                ],
                dtype=torch.float32,
                device=device,
            )

            with dllm_nvtx_range(f"dllm_focus_step{step}"):
                logits_output, retained_maps, new_lens = self._focus_reduced_forward(
                    model_runner, forward_batch, avg_decoded
                )

            dllm_nvtx_push(f"dllm_focus_commit.step{step}")
            self._commit_step(
                forward_batch,
                logits_output.full_logits,
                retained_maps,
                new_lens,
                token_sum,
                total_steps,
            )
            dllm_nvtx_pop()

        # Final full forward repopulates the whole-block KV for the next block's
        # context (the reduced steps only wrote retained KV at L>=2).
        with dllm_nvtx_range("dllm_focus_final_forward"):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        return logits_output, next_token_ids_list, can_run_cuda_graph


# Register algorithm
Algorithm = Focus
