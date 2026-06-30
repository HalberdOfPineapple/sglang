"""FOCUS reduced-forward helpers: state compaction + attention-metadata rebuild.

Phase A: PyTorch. These build the inputs for the *suffix* (layers 1-attn..L run
on the retained set ``|S| ≪ B``). Mirrors the official ``focus_compact_states``
(kernels/cuda/focus.py:676) which fuses the per-token gathers into one Triton
kernel; here the gather is a plain ``index_select`` over a flat batch, which is
also the natural numerical oracle.

The suffix's attention-KV read (sparse ragged vs compacted) is handled
separately; this module only produces the compacted per-token tensors and the
new ragged boundaries the suffix needs.
"""

from typing import Dict, List, Optional, Tuple

import torch


def build_retained_index(
    retained_maps: List[torch.Tensor],
    seq_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global gather index + per-request retained counts.

    Args:
        retained_maps: per-request sorted retained block-local indices (output of
            ``select_and_enforce_constraints``).
        seq_offsets: [batch+1] CSR boundaries of the processing set within the
            flat ``[total_tokens, ...]`` batch.

    Returns:
        keep_index: [total_retained] long, gather index into the flat batch
            (request-major, ascending within each request).
        new_lens: [batch] int32, retained token count per request (the new
            ``extend_seq_lens`` for the suffix).
    """
    device = seq_offsets.device
    parts: List[torch.Tensor] = []
    new_lens: List[int] = []
    for b, rm in enumerate(retained_maps):
        base = int(seq_offsets[b].item())
        parts.append(rm.to(torch.long).to(device) + base)
        new_lens.append(int(rm.numel()))
    keep_index = (
        torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)
    )
    return keep_index, torch.tensor(new_lens, dtype=torch.int32, device=device)


def build_retained_index_from_mask(
    retain_mask_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """On-device (§A2) equivalent of ``build_retained_index`` — zero D2H sync.

    The processing set is uniform ``block_size`` per request (Phase A; DC+ not yet
    behaviorally active), so the per-request retain masks stack into a dense
    ``[bs, block_size]`` bool grid. The request-major flat ``nonzero`` then gives
    exactly the same ``keep_index`` the Python-loop oracle builds (row b's flat
    index ``b*block_size + pos`` equals ``seq_offsets[b] + pos``), and the row sum
    gives ``new_lens`` — both without a single ``.item()``.

    Args:
        retain_mask_2d: [bs, block_size] bool, True = retained block position
            (non-masked processing positions + selected masked positions).

    Returns:
        keep_index: [total_retained] long gather index into the flat batch,
            request-major and ascending within each request.
        new_lens: [bs] int32 retained token count per request.
    """
    keep_index = retain_mask_2d.reshape(-1).nonzero(as_tuple=True)[0]
    new_lens = retain_mask_2d.sum(dim=1).to(torch.int32)
    return keep_index, new_lens


def focus_compact_states(
    keep_index: torch.Tensor,
    tensors: Dict[str, Optional[torch.Tensor]],
) -> Dict[str, Optional[torch.Tensor]]:
    """Gather each ``[total_tokens, ...]`` tensor down to the retained tokens.

    ``None`` entries pass through. Equivalent to the official fused compaction;
    correctness oracle is ``t[keep_index]`` itself, so the unit test builds an
    independent double-loop reference for the index construction.
    """
    out: Dict[str, Optional[torch.Tensor]] = {}
    for name, t in tensors.items():
        if t is None:
            out[name] = None
        else:
            out[name] = t.index_select(0, keep_index.to(t.device))
    return out


def cu_seqlens_from_lens(lens: torch.Tensor) -> torch.Tensor:
    """Exclusive-prefix-sum boundaries [batch+1] from per-request lengths."""
    out = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=lens.device)
    out[1:] = torch.cumsum(lens.to(torch.int64), dim=0).to(torch.int32)
    return out
