"""FOCUS §C — Phase-S CUDA-graph runner (lazy capture of forward_focus_rest_and_logits).

Captures Phase S (L2..L + norm + lm_head on the retained |S| set, the ~62% wall
lever) as a CUDA graph keyed on ``(bs, bucket)`` where ``bucket`` rounds Σ|S| up the
``focus_graph`` ladder. Capture is **lazy**: the first time a ``(bs, bucket)`` is
seen during serving, the real (padded) Phase-S forward is captured using the live
``fb_s`` — so the captured run itself produces the correct result for that step.
Thereafter the graph is replayed, collapsing ~18 MoE layers' worth of eager kernel
launches into one replay.

Mechanism (validated piecewise by ``test_focus_phase_s_graph_gpu.py``): a per-bs
``BatchPrefillWithPagedKVCacheWrapper(use_cuda_graph=True)`` with static metadata
buffers is injected into ``attn_backend.forward_metadata`` for the duration of the
capture/replay; the model's L2..L attention reads it (``forward_extend`` ->
``forward_metadata.prefill_wrappers[idx]``). ``plan()`` runs on the host (outside the
graph); the attention ``forward`` runs inside it.

Padding (Option B): FlashInfer locks a cuda-graph wrapper's batch size (= ragged
segment count) at construction, so we keep ONE wrapper per bs and fold the
``bucket − Σ|S|`` pad tokens into the LAST real request's query segment (segment
count stays == bs; no empty segment when pad_len==0). Pad rows attend the last
request's real KV (non-causal ⇒ valid; their logits are sliced off) and write their
own L≥2 KV to a **non-retained block slot the batch already owns** (a block position
≥ |S_b|, never read by Phase S, overwritten by the block's final full forward) — so
padded compute cannot corrupt real KV. NO allocator ``alloc()`` (that trips SGLang's
pool memory-leak detector).

Gated by ``SGLANG_FOCUS_GRAPH=1`` (default OFF). ANY capture/replay failure falls
back to eager (``run`` returns ``None``), so the working eager FOCUS path and normal
serving are never at risk.
"""

import logging
from typing import Optional

import torch

from sglang.srt.dllm.algorithm.focus_graph import (
    build_capture_token_buckets,
    phase_s_token_bucket,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

logger = logging.getLogger(__name__)


class FocusPhaseSGraphRunner:
    def __init__(self, model_runner, block_size: int):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.mr = model_runner
        self.model = model_runner.model
        self.attn = model_runner.attn_backend
        self.device = model_runner.device
        self.block_size = block_size
        self.dtype = model_runner.dtype

        cfg = model_runner.model_config
        self.hidden = cfg.hidden_size
        max_bs = model_runner.server_args.max_running_requests or 16
        self.max_bs = int(max_bs)
        self.T_max = self.max_bs * block_size
        self.buckets = build_capture_token_buckets(self.T_max)

        # No KV slot is reserved from the allocator (that trips SGLang's pool
        # memory-leak detector). Instead, when padding is needed (pad_len>0) there
        # is ALWAYS a non-retained block slot the batch already owns — Phase S uses
        # only Σ|S| of the bs·B block slots, and the leftover ones get overwritten
        # by the block's final full forward — so pad tokens write their (never-read)
        # KV there. The scratch slot is picked per step from ``base_out_cache_loc``.
        T = self.T_max
        self.hidden_buf = torch.zeros(T, self.hidden, dtype=self.dtype, device=self.device)
        self.residual_buf = torch.zeros(T, self.hidden, dtype=self.dtype, device=self.device)
        self.input_ids_buf = torch.zeros(T, dtype=torch.int64, device=self.device)
        self.positions_buf = torch.zeros(T, dtype=torch.int64, device=self.device)
        self.out_cache_loc_buf = torch.zeros(T, dtype=torch.int64, device=self.device)

        # Head/dtype params live on the prefill indices-updater (not the backend);
        # max_context_len from model_config. Sourced once here.
        iup = self.attn.indices_updater_prefill
        self.num_qo_heads = iup.num_qo_heads
        self.num_kv_heads = iup.num_kv_heads
        self.head_dim = iup.head_dim
        self.kv_data_type = iup.data_type
        # Cap the per-request kv capacity so the priming plan + kv_indices buffer
        # stay modest; real contexts here are far smaller. Clamp to the model ctx.
        self.max_context_len = min(int(cfg.context_len), 8192)
        max_context_len = self.max_context_len

        # Static Phase-S FlashInfer metadata buffers. The pad tokens are folded
        # into the LAST real request's segment (not a separate segment), so the
        # ragged layout has exactly ``bs`` segments — this keeps the FlashInfer
        # cuda-graph wrapper's batch size fixed per bs (it is locked at wrapper
        # construction) and avoids any empty-segment edge case when pad_len==0.
        self._wrapper_cls = BatchPrefillWithPagedKVCacheWrapper
        self.qo_indptr = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=self.device)
        self.kv_indptr = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=self.device)
        max_kv = T + self.max_bs * max_context_len
        self.kv_indices = torch.zeros(max_kv, dtype=torch.int32, device=self.device)
        self.kv_last_page = torch.ones(self.max_bs, dtype=torch.int32, device=self.device)
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.wrappers = {}    # (bs, bucket) -> wrapper (qo total fixed at bucket)
        self.graphs = {}      # (bs, bucket) -> CUDAGraph
        self.logits_out = {}  # (bs, bucket) -> full_logits static tensor
        self.cap_kv = {}      # (bs, bucket) -> kv capacity captured (re-capture if exceeded)
        self._failed = False  # one hard failure disables the runner for the session
        logger.info(
            f"[focus-graph] runner ready: max_bs={self.max_bs} T_max={self.T_max} "
            f"buckets={self.buckets}"
        )

    # ------------------------------------------------------------------ metadata
    def _get_wrapper(self, bs: int, bucket: int):
        """Lazily build a cuda-graph prefill wrapper keyed by ``(bs, bucket)``.

        FlashInfer locks BOTH the batch size (= number of ragged segments == bs) AND
        the total qo-row count at the wrapper's FIRST plan, so each distinct
        ``(bs, bucket)`` needs its own wrapper (qo total == bucket, fixed). The kv
        length is also bounded by the first plan; since context accumulates across
        blocks, a later step whose kv exceeds the first plan's capacity soft-falls
        back to eager for that step (caught in ``run``, no permanent disable).
        """
        w = self.wrappers.get((bs, bucket))
        if w is None:
            w = self._wrapper_cls(
                self.attn.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self.qo_indptr[: bs + 1],
                paged_kv_indptr_buf=self.kv_indptr[: bs + 1],
                paged_kv_indices_buf=self.kv_indices,
                paged_kv_last_page_len_buf=self.kv_last_page[:bs],
            )
            self.wrappers[(bs, bucket)] = w
        return w

    def _plan(self, fb_s, new_lens, new_lens_cpu, bs: int, bucket: int):
        """Fill the static metadata buffers (Option B: pad folded into last segment).

        Layout has exactly ``bs`` segments. The pad tokens (``bucket − Σ|S|``) are
        appended to the LAST real request's query segment: its qo_len becomes
        |S_{bs-1}|+pad_len while its kv (context+|S_{bs-1}|) is unchanged — the pad
        rows attend the last request's real KV (non-causal; their logits are sliced
        off) and write their own KV to scratch via ``out_cache_loc``. So kv_indices
        carries ONLY the real per-request slices (no pad kv). ``fb_s.seq_lens`` is
        context+|S| per req, so each KV slice is ``req_to_token[req,0:context+|S|]``,
        built by the same Triton kernel the backend uses. Returns the planned wrapper
        or None if the layout doesn't fit.
        """
        real_tokens = int(new_lens_cpu.sum())
        pad_len = bucket - real_tokens
        if pad_len < 0:
            return None
        seq_lens = fb_s.seq_lens[:bs].to(torch.int32)  # context+|S| per req
        real_kv = int(fb_s.seq_lens_cpu[:bs].sum())
        if real_kv > self.kv_indices.numel():
            return None

        # kv_indptr: cumsum of context+|S| (unchanged by padding).
        kv_cumsum = torch.cumsum(seq_lens.to(torch.int64), dim=0)
        self.kv_indptr.zero_()
        self.kv_indptr[1 : bs + 1].copy_(kv_cumsum.to(torch.int32))
        # qo_indptr: cumsum of |S|, with pad_len added to the LAST segment only.
        qo_lens = new_lens[:bs].to(torch.int64).clone()
        qo_lens[-1] += pad_len
        qo_cumsum = torch.cumsum(qo_lens, dim=0)
        self.qo_indptr.zero_()
        self.qo_indptr[1 : bs + 1].copy_(qo_cumsum.to(torch.int32))  # last == bucket

        # kv_indices for the bs real requests (context+|S| slice of req_to_token).
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            fb_s.req_pool_indices[:bs],
            seq_lens,
            self.kv_indptr[: bs + 1],
            None,
            self.kv_indices,
            self.req_to_token.shape[1],
        )
        self.kv_last_page[:bs].fill_(1)

        w = self._get_wrapper(bs, bucket)
        w.plan(
            self.qo_indptr[: bs + 1],
            self.kv_indptr[: bs + 1],
            self.kv_indices[:real_kv],
            self.kv_last_page[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,  # page_size
            causal=False,
            q_data_type=self.dtype,
            kv_data_type=self.kv_data_type,
        )
        return w

    # ------------------------------------------------------------------ run/capture
    def run(
        self,
        fb_s,
        hidden_s: torch.Tensor,
        input_ids_s: torch.Tensor,
        positions_s: torch.Tensor,
        residual_s: Optional[torch.Tensor],
        new_lens: torch.Tensor,
        new_lens_cpu: torch.Tensor,
        base_out_cache_loc: torch.Tensor,
    ):
        """Run Phase S via a (lazily-captured) CUDA graph. Returns full_logits over
        the real Σ|S| rows, or ``None`` to signal the caller to fall back to eager.

        ``base_out_cache_loc`` is the full-block ``out_cache_loc`` ([bs·B]); its
        non-retained slots provide the per-step scratch target for pad KV writes.
        """
        if self._failed:
            return None
        bs = fb_s.batch_size
        total_s = int(hidden_s.shape[0])
        if total_s <= 0 or total_s > self.T_max or bs > self.max_bs:
            return None
        bucket = phase_s_token_bucket(total_s, self.T_max)
        pad_len = bucket - total_s
        key = (bs, bucket)
        # kv (context+|S|) grows across blocks; a captured graph fixes the kv
        # capacity at its first plan, so re-capture this (bs,bucket) when the live
        # kv exceeds what we captured (context grew). Equal/smaller ⇒ replay.
        real_kv = int(fb_s.seq_lens_cpu[:bs].sum())
        need_capture = key not in self.graphs or real_kv > self.cap_kv.get(key, -1)
        if need_capture and key in self.graphs:
            self.graphs.pop(key, None)
            self.logits_out.pop(key, None)
            self.wrappers.pop(key, None)

        # Everything below can mutate shared backend state — wrap it all so ANY
        # failure (plan, capture, replay) disables the runner and falls back to
        # eager, with the backend metadata / fb_s always restored.
        saved_meta = self.attn.forward_metadata
        saved_oc = fb_s.out_cache_loc
        from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata

        try:
            # Pad scratch = a non-retained block slot (device scalar, no sync).
            # Exists whenever pad_len>0 since Σ|S| < bucket ≤ bs·B ⇒ a slot is unused.
            scratch = None
            if pad_len > 0:
                base2d = base_out_cache_loc[: bs * self.block_size].view(
                    bs, self.block_size
                )
                nonret = base2d[
                    torch.arange(self.block_size, device=self.device)
                    >= new_lens[:bs].unsqueeze(1)
                ]
                if nonret.numel() == 0:
                    return None  # shouldn't happen; fall back to eager
                scratch = nonret[0]

            # Load real data into the static buffers; pad range -> zeros / scratch.
            self.hidden_buf[:total_s].copy_(hidden_s)
            self.hidden_buf[total_s:bucket].zero_()
            if residual_s is not None:
                self.residual_buf[:total_s].copy_(residual_s)
                self.residual_buf[total_s:bucket].zero_()
            self.input_ids_buf[:total_s].copy_(input_ids_s)
            self.input_ids_buf[total_s:bucket].zero_()
            self.positions_buf[:total_s].copy_(positions_s)
            self.positions_buf[total_s:bucket].zero_()
            self.out_cache_loc_buf[:total_s].copy_(fb_s.out_cache_loc)
            if pad_len > 0:
                self.out_cache_loc_buf[total_s:bucket] = scratch

            wrapper = self._plan(fb_s, new_lens, new_lens_cpu, bs, bucket)
            if wrapper is None:
                return None

            self.attn.forward_metadata = PrefillMetadata([wrapper], False, False)
            fb_s.out_cache_loc = self.out_cache_loc_buf[:bucket]
            res_arg = self.residual_buf[:bucket] if residual_s is not None else None
            if need_capture:
                self._capture(key, fb_s, bucket, res_arg)
                self.cap_kv[key] = real_kv
            else:
                self.graphs[key].replay()
            return self.logits_out[key][:total_s]
        except Exception as e:  # never break serving — disable + fall back to eager
            logger.warning(f"[focus-graph] disabled after failure on {key}: {e!r}")
            self._failed = True
            self.graphs.pop(key, None)
            self.logits_out.pop(key, None)
            return None
        finally:
            self.attn.forward_metadata = saved_meta
            fb_s.out_cache_loc = saved_oc

    def _capture(self, key, fb_s, bucket: int, res_arg):
        h = self.hidden_buf[:bucket]
        ids = self.input_ids_buf[:bucket]
        pos = self.positions_buf[:bucket]
        # Warmup the exact static-buffer call (also produces this step's real result).
        for _ in range(2):
            self.model.forward_focus_rest_and_logits(h, ids, pos, fb_s, res_arg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model.forward_focus_rest_and_logits(h, ids, pos, fb_s, res_arg)
        self.graphs[key] = g
        self.logits_out[key] = out.full_logits
        logger.info(f"[focus-graph] captured Phase-S graph {key}")
