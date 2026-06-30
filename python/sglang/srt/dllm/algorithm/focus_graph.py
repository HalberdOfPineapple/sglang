"""FOCUS §C — Phase-S CUDA-graph foundation (bucketization + ragged pad layout).

Phase S (``forward_focus_rest_and_logits`` = L2..L + norm + lm_head on the retained
|S| set) is the ~62% of FOCUS wall (measured F3-lite). A CUDA graph is a fixed-shape
recording, so to capture Phase S we **bucket** the data-dependent total token count
Σ|S| up to a captured size and **pad** the real tokens to that bucket, then rewrite
the ragged FlashInfer metadata at replay. This module is the pure-tensor foundation
(bucket ladder + padded ragged layout) the graph capture/replay will consume — no
CUDA / model / FlashInfer dependency, so it is unit-testable in isolation. See
``notes/focus_phase_s_graph_design.md`` for the full design.

Correctness invariant (the most important one in §C): **padded tokens must write
their KV to a reserved scratch slot**, never a real block slot, so padded compute
cannot corrupt the block KV. ``build_phase_s_graph_layout`` therefore stamps the
pad range of ``out_cache_loc`` with ``scratch_loc`` and routes the trailing pad
segment's KV to scratch.
"""

from dataclasses import dataclass
from typing import List

import torch


def phase_s_token_bucket(n: int, max_tokens: int) -> int:
    """Smallest captured-graph token bucket ≥ ``n`` (mirrors official ladder).

    Ladder = powers of 2 up to 256, then stride-256 above 256, capped at
    ``max_tokens`` (= max_bs · block_size). ``n ≤ 0`` rounds to 1 (a graph always
    runs ≥ 1 token). The net Phase-S win requires ``bucket(Σ|S|) ≪ bs·B``; at high
    redundancy the bucket approaches bs·B and the pad eats the saving.
    """
    if n <= 1:
        return min(1, max_tokens)
    if n >= max_tokens:
        return max_tokens
    if n <= 256:
        b = 1
        while b < n:
            b <<= 1
        return min(b, max_tokens)
    b = ((n + 255) // 256) * 256
    return min(b, max_tokens)


def build_capture_token_buckets(max_tokens: int) -> List[int]:
    """All buckets to capture for a max token count (the distinct ladder rungs)."""
    buckets: List[int] = []
    b = 1
    while b <= min(256, max_tokens):
        buckets.append(b)
        b <<= 1
    b = 512
    while b <= max_tokens:
        buckets.append(b)
        b += 256
    if not buckets or buckets[-1] != max_tokens:
        buckets.append(max_tokens)
    return sorted(set(buckets))


@dataclass
class PhaseSGraphLayout:
    """Padded ragged Phase-S layout for one replay (host-side ints/tensors).

    ``qo_lens`` / ``kv_lens`` have ``bs + 1`` entries: the first ``bs`` are the real
    requests (qo=|S_b|, kv=context_b+|S_b|), the last is the synthetic **pad
    segment** (qo=kv=``pad_len``) that absorbs ``bucket − Σ|S|`` padded tokens and
    attends only scratch KV. ``real_tokens`` = Σ|S| (the prefix of the token buffer
    holding live logits); ``bucket`` = the captured token count.
    """

    bucket: int
    real_tokens: int
    pad_len: int
    qo_lens: torch.Tensor  # [bs + 1] int64 (last = pad segment)
    kv_lens: torch.Tensor  # [bs + 1] int64 (last = pad segment)
    seg_is_pad: torch.Tensor  # [bs + 1] bool (only last True)


def build_phase_s_graph_layout(
    new_lens_cpu: torch.Tensor,
    context_lens_cpu: torch.Tensor,
    bucket: int,
) -> PhaseSGraphLayout:
    """Ragged qo/kv segment lengths for a Phase-S graph replay at a fixed bucket.

    Args:
        new_lens_cpu: [bs] int retained |S_b| per request (host; the §A3 D2H).
        context_lens_cpu: [bs] int context length per request (= original
            ``seq_lens − block_size``), host.
        bucket: captured token count (≥ Σ|S_b|), from ``phase_s_token_bucket``.

    Returns:
        PhaseSGraphLayout with the real requests + one trailing pad segment whose
        qo=kv=``pad_len`` (attends scratch). ``Σ qo_lens == bucket`` by construction.
    """
    new_lens_cpu = new_lens_cpu.to(torch.int64)
    context_lens_cpu = context_lens_cpu.to(torch.int64)
    real_tokens = int(new_lens_cpu.sum())
    if real_tokens > bucket:
        raise ValueError(f"Σ|S|={real_tokens} exceeds bucket={bucket}")
    pad_len = bucket - real_tokens

    pad_t = torch.tensor([pad_len], dtype=torch.int64)
    qo_lens = torch.cat([new_lens_cpu, pad_t])
    kv_lens = torch.cat([context_lens_cpu + new_lens_cpu, pad_t])
    seg_is_pad = torch.zeros(new_lens_cpu.numel() + 1, dtype=torch.bool)
    seg_is_pad[-1] = True
    return PhaseSGraphLayout(
        bucket=bucket,
        real_tokens=real_tokens,
        pad_len=pad_len,
        qo_lens=qo_lens,
        kv_lens=kv_lens,
        seg_is_pad=seg_is_pad,
    )


def pad_phase_s_out_cache_loc(
    real_out_cache_loc: torch.Tensor,
    bucket: int,
    scratch_loc: int,
) -> torch.Tensor:
    """Pad the block-prefix Phase-S ``out_cache_loc`` to ``bucket`` with scratch.

    Real retained tokens keep their block-prefix KV slots (from
    ``focus_forward.build_phase_s_out_cache_loc``); the ``bucket − Σ|S|`` padded
    tokens all write to ``scratch_loc`` so padded compute cannot corrupt real KV.
    """
    real_tokens = real_out_cache_loc.numel()
    if real_tokens > bucket:
        raise ValueError(f"out_cache_loc len {real_tokens} exceeds bucket {bucket}")
    out = torch.full(
        (bucket,),
        scratch_loc,
        dtype=real_out_cache_loc.dtype,
        device=real_out_cache_loc.device,
    )
    out[:real_tokens] = real_out_cache_loc
    return out


def pad_phase_s_tokens(
    t: torch.Tensor,
    bucket: int,
    pad_value=0,
) -> torch.Tensor:
    """Pad a per-token tensor ``[Σ|S|, ...]`` up to ``[bucket, ...]`` with ``pad_value``.

    Used for ``input_ids``/``positions`` (and, with ``pad_value=0``,
    ``hidden``/``residual``). The padded rows compute but their logits are sliced
    off after replay; their KV writes go to scratch (see
    ``pad_phase_s_out_cache_loc``).
    """
    real_tokens = t.shape[0]
    if real_tokens > bucket:
        raise ValueError(f"token dim {real_tokens} exceeds bucket {bucket}")
    if real_tokens == bucket:
        return t
    pad_shape = (bucket - real_tokens,) + tuple(t.shape[1:])
    pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)
