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
import time
from collections import defaultdict
from contextlib import contextmanager
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
from sglang.srt.dllm.algorithm.focus_reduce import build_retained_index_from_mask
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
        # SGLANG_FOCUS_PHASE_TIMING=1 attributes per-step wall time to host vs
        # device phases (P/A1/S metadata + forward, selection, commit) to decide
        # where §A5/§B/§C should aim. Syncs at phase boundaries (serializes, so
        # absolute ms are inflated, but the *relative* split is faithful). No-op
        # when unset.
        self._phase_timing = os.environ.get("SGLANG_FOCUS_PHASE_TIMING") == "1"
        self._phase_acc = defaultdict(float)
        self._phase_steps = 0
        # SGLANG_FOCUS_GRAPH=1 replays a CUDA graph for Phase S (the ~62% lever);
        # default OFF. Lazily built on first run; any failure falls back to eager.
        self._graph_enabled = os.environ.get("SGLANG_FOCUS_GRAPH") == "1"
        self._graph_runner = None

    def _ensure_graph_runner(self, model_runner):
        if self._graph_runner is not None or not self._graph_enabled:
            return
        try:
            from sglang.srt.dllm.algorithm.focus_graph_runner import (
                FocusPhaseSGraphRunner,
            )

            self._graph_runner = FocusPhaseSGraphRunner(model_runner, self.block_size)
        except Exception as e:  # stay eager if the runner can't be built
            print(f"[focus-graph] init failed, staying eager: {e!r}", flush=True)
            self._graph_enabled = False

    @contextmanager
    def _phase(self, label: str):
        """Wall-clock a code region into ``self._phase_acc[label]`` (ms)."""
        if not self._phase_timing:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self._phase_acc[label] += (time.perf_counter() - t0) * 1e3

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
    ) -> torch.Tensor:
        """
        ΔI → budget (no N_σ) → should_evict → enforced retained set S per req.
        Inputs:
            focus_view: CSR view of the full-block prefix forward.
            forward_batch: The forward batch containing input data.
        Outputs:
            retain_mask_2d: [batch_size, block_size] bool — True at retained block
            positions (non-masked processing positions + selected masked ones).
        """
        delta_I = focus_view.get_delta_importance()  # [batch * block_size]
        mask = forward_batch.input_ids == self.mask_id
        seq_offsets = focus_view.seq_offsets
        batch_size = focus_view.batch_size
        # §A1: the prefix processing set is a uniform block_size per request, so
        # mask_lengths is a single reshape+row-sum — no per-request .item() loop.
        mask_lengths = mask.view(batch_size, self.block_size).sum(dim=1).to(torch.int32)
        targets = compute_focus_targets(
            mask_lengths, focus_view.avg_decoded, self.alpha
        )
        should_evict = compute_should_evict(mask_lengths, targets)
        retain_masks, _ = select_and_enforce_constraints(
            delta_I,
            mask,
            seq_offsets,
            targets,
            should_evict,
            self.block_size,
        )
        # Uniform block_length ⇒ stack the per-request [block_size] masks into a
        # dense [bs, block_size] grid for the on-device index build (§A2).
        return torch.stack(retain_masks)

    def _focus_reduced_forward(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        avg_decoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One reduced denoising forward.

        Returns ``(full_logits, keep_index, new_lens, new_lens_cpu)``.
        ``full_logits`` is [Σ|S_b|, vocab], request-major; row j maps to the flat
        block position ``keep_index[j]`` (∈ [0, bs*block_size)). Phase S runs either
        eager or via a replayed CUDA graph (§C, ``SGLANG_FOCUS_GRAPH=1``) — both
        yield the same ``full_logits`` tensor. ``new_lens`` is the per-request
        retained count (device), ``new_lens_cpu`` its host copy (the single per-step
        D2H), reused by metadata builders and redundancy logs.
        """
        model = model_runner.model
        attn_backend = model_runner.attn_backend
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        # --- Phase P: full-block prefix (L0 full + L1 QKV+RoPE+fill). ---
        focus_view = self._build_focus_view(batch_size, device, avg_decoded)
        forward_batch.focus_view = focus_view
        with self._phase("p_meta"):
            attn_backend.init_forward_metadata(forward_batch)
        with dllm_nvtx_range("focus_prefix"), self._phase("p_fwd"):
            q1, residual = model.forward_focus_prefix(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
        forward_batch.focus_view = None

        # --- host: selection + compaction. ---
        with self._phase("select"):
            retain_mask_2d = self._select_retained(focus_view, forward_batch)
            keep_index, new_lens = build_retained_index_from_mask(retain_mask_2d)
            # The ONE D2H sync per step (§A3): all host-side phase-metadata lists
            # are deterministic host arithmetic on this + the base seq_lens_cpu.
            new_lens_cpu = new_lens.detach().cpu()
            q1_s = q1.index_select(0, keep_index)
            residual_s = (
                None if residual is None else residual.index_select(0, keep_index)
            )
            positions_s = forward_batch.positions.index_select(0, keep_index)
            input_ids_s = forward_batch.input_ids.index_select(0, keep_index)

        # --- Phase A1: L1 attention on |S| vs full-block KV + L1 MLP. ---
        with self._phase("a1_meta"):
            fb_a1 = make_focus_phase_batch(
                forward_batch,
                PHASE_A1,
                self.block_size,
                new_lens,
                input_ids_s,
                positions_s,
                new_lens_cpu=new_lens_cpu,
            )
            attn_backend.init_forward_metadata(fb_a1)
        with dllm_nvtx_range("focus_l1_attn"), self._phase("a1_fwd"):
            hidden_s, residual_s = model.forward_focus_l1_suffix(
                q1_s, fb_a1, residual_s
            )

        # --- Phase S: L2..L on |S| (KV → block prefix, read context+|S|). ---
        with self._phase("s_meta"):
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
                new_lens_cpu=new_lens_cpu,
            )

        # §C: replay a CUDA graph for Phase S when enabled; else eager. The graph
        # path returns full_logits over the real Σ|S| rows; None ⇒ eager fallback.
        full_logits = None
        if self._graph_runner is not None:
            with dllm_nvtx_range("focus_suffix_graph"), self._phase("s_fwd"):
                full_logits = self._graph_runner.run(
                    fb_s,
                    hidden_s,
                    input_ids_s,
                    positions_s,
                    residual_s,
                    new_lens,
                    new_lens_cpu,
                    forward_batch.out_cache_loc,
                )
        if full_logits is None:
            with self._phase("s_meta"):
                attn_backend.init_forward_metadata(fb_s)
            with dllm_nvtx_range("focus_suffix"), self._phase("s_fwd"):
                logits_output = model.forward_focus_rest_and_logits(
                    hidden_s, input_ids_s, positions_s, fb_s, residual_s
                )
            full_logits = logits_output.full_logits

        if self._log_redundancy or self._redundancy_csv:
            total = self.block_size * batch_size
            kept = int(new_lens_cpu.sum())  # host sum, reuses the §A3 D2H
            ratio = kept / total
            if self._log_redundancy:
                print(
                    f"[focus] |S|/(B*bs) = {kept}/{total} = {ratio:.3f} "
                    f"(per-req |S|={new_lens_cpu.tolist()})",
                    flush=True,
                )
            if self._redundancy_csv:
                fh = self._redundancy_fh()
                if fh is not None:
                    fh.write(
                        f"{batch_size},{kept},{total},{ratio:.4f},"
                        f"\"{new_lens_cpu.tolist()}\"\n"
                    )
                    fh.flush()
        return full_logits, keep_index, new_lens, new_lens_cpu

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
        keep_index: torch.Tensor,
        token_sum: torch.Tensor,
        total_steps: torch.Tensor,
    ):
        """
        Decode the |S| retained logits and commit the confident masked ones.

        Only retained masked positions are eligible to decode this step (the
        evicted ones never produced suffix logits); each block commits all
        positions above ``threshold`` or, failing that, its single most-confident
        retained masked position (progress guarantee).

        §A4 — fully ragged/vectorized, no per-request Python loop and no ``.item()``:
        the |S|-flat logits scatter (via ``keep_index``) into dense ``[bs, B]``
        confidence / predicted-token grids, then the commit decision and the write
        are batched tensor ops. ``token_sum`` / ``total_steps`` are device int64
        accumulators updated in place (drive the next step's budget on-device).
        """
        bs = forward_batch.batch_size
        B = self.block_size
        device = forward_batch.input_ids.device
        input_ids = forward_batch.input_ids  # [bs*B], mutated in place via view
        block_mask_flat = input_ids == self.mask_id

        x_ret = torch.argmax(full_logits_s, dim=-1)  # [Σ|S|]
        p_ret = torch.gather(
            F.softmax(full_logits_s, dim=-1), -1, x_ret.unsqueeze(-1)
        ).squeeze(-1)

        # Scatter retained results back into request-major flat [bs*B] buffers.
        # Only retained *masked* positions get a finite confidence; everything
        # else stays −inf, exactly mirroring the per-request reference.
        # ``confidence`` is float32 (mirrors the per-request reference, whose
        # default-dtype −inf buffer upcast the bf16 probabilities) so the
        # ``> threshold`` decision is taken in the same precision — avoids a
        # bf16-vs-fp32 flip on borderline commits.
        ret_masked = block_mask_flat[keep_index]
        p_ret32 = p_ret.to(torch.float32)
        neg_inf = torch.full_like(p_ret32, -np.inf)
        conf_flat = torch.full((bs * B,), -np.inf, device=device, dtype=torch.float32)
        conf_flat[keep_index] = torch.where(ret_masked, p_ret32, neg_inf)
        x_pred_flat = input_ids.clone()
        x_pred_flat[keep_index] = torch.where(
            ret_masked, x_ret.to(input_ids.dtype), input_ids[keep_index]
        )

        conf = conf_flat.view(bs, B)
        x_pred = x_pred_flat.view(bs, B)
        transfer = conf > self.threshold  # [bs, B]
        # Progress guarantee: a request with a decodable (finite-confidence)
        # retained-masked position but nothing above threshold commits its single
        # most-confident one. ``conf == row_max`` selects that argmax without a
        # data-dependent ``.item()`` sync (ties are astronomically unlikely).
        decodable = torch.isfinite(conf).any(dim=1)  # [bs]
        need_top1 = (~transfer.any(dim=1)) & decodable
        row_max = conf.max(dim=1, keepdim=True).values
        transfer = transfer | (need_top1.unsqueeze(1) & (conf == row_max))

        input_ids.view(bs, B)[transfer] = x_pred[transfer]
        token_sum.add_(transfer.sum(dim=1).to(token_sum.dtype))
        total_steps.add_(transfer.any(dim=1).to(total_steps.dtype))

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
        self._ensure_graph_runner(model_runner)

        mask_index = forward_batch.input_ids == self.mask_id
        if torch.sum(mask_index).item() == 0:
            with dllm_nvtx_range("dllm_prefill_forward"):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # start = #already-committed positions per block (prompt prefix inside
        # block). One D2H for the whole batch (used only to slice the final output).
        mask_counts = (
            (forward_batch.input_ids.view(batch_size, self.block_size) == self.mask_id)
            .sum(dim=1)
        )
        start_list = (self.block_size - mask_counts).tolist()

        # Per-request cumulative decode statistics drive the budget K (Eq. 17-19).
        # Kept on-device (§A4) so commit/budget never round-trip to host.
        token_sum = torch.zeros(batch_size, dtype=torch.int64, device=device)
        total_steps = torch.zeros(batch_size, dtype=torch.int64, device=device)

        for step in range(self.block_size):
            if torch.sum(forward_batch.input_ids == self.mask_id).item() == 0:
                break

            # N̄_decoded per request (cumulative-mean yield); default n_bar_init
            # before the first commit. Pure on-device arithmetic.
            avg_decoded = torch.where(
                total_steps > 0,
                token_sum.to(torch.float32) / total_steps.clamp(min=1).to(torch.float32),
                torch.full((batch_size,), self.n_bar_init, device=device),
            )

            with dllm_nvtx_range(f"dllm_focus_step{step}"):
                full_logits, keep_index, _, _ = self._focus_reduced_forward(
                    model_runner, forward_batch, avg_decoded
                )

            dllm_nvtx_push(f"dllm_focus_commit.step{step}")
            with self._phase("commit"):
                self._commit_step(
                    forward_batch,
                    full_logits,
                    keep_index,
                    token_sum,
                    total_steps,
                )
            dllm_nvtx_pop()
            self._phase_steps += 1

        if self._phase_timing and self._phase_steps:
            total = sum(self._phase_acc.values()) or 1.0
            parts = " ".join(
                f"{k}={v:.0f}({100 * v / total:.0f}%)"
                for k, v in sorted(self._phase_acc.items(), key=lambda x: -x[1])
            )
            print(
                f"[focus-timing] bs={batch_size} steps={self._phase_steps} "
                f"total={total:.0f}ms :: {parts}",
                flush=True,
            )
            self._phase_acc.clear()
            self._phase_steps = 0

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
