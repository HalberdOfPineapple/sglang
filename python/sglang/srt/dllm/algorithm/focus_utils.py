"""FOCUS helper functions for importance, budgeting, and selection.

Phase A: PyTorch reference implementations.
Phase B: Will be replaced with Triton kernels.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FocusRuntimeView:
    """Runtime state threaded through the forward pass for FOCUS.

    Lives on ``forward_batch.focus_view``. The attention layers read
    ``seq_offsets``/``maxpool_k``/``importance_layers`` to collect per-layer
    importance via ``set_layer_importance``; the algorithm reads ``importance``
    back after the forward to compute ΔI and run selection.

    For Phase A the block layout is uniform (every request contributes a full
    ``block_size`` slice), so ``seq_offsets`` is a simple arange. Once the
    delayed cache shrinks the processed set this becomes a true ragged CSR.
    """

    block_size: int
    batch_size: int
    seq_offsets: torch.Tensor  # [batch_size + 1] CSR boundaries over processed tokens
    maxpool_k: int = 3
    importance_layers: Tuple[int, ...] = (0, 1)
    # avg_decoded[b] = N̄_decoded for request b (cumulative-mean decode yield)
    avg_decoded: Optional[torch.Tensor] = None
    # Filled in by attention layers, keyed by layer_id.
    importance: dict = field(default_factory=dict)

    def set_layer_importance(self, layer_id: int, value: torch.Tensor):
        self.importance[layer_id] = value

    def get_delta_importance(self) -> torch.Tensor:
        """ΔI = I^(Layer1) − I^(Layer0) (Eq. 3), the decodability signal."""
        l0, l1 = self.importance_layers[0], self.importance_layers[1]
        return self.importance[l1] - self.importance[l0]

    def has_importance(self) -> bool:
        return all(layer in self.importance for layer in self.importance_layers)


def compute_importance_side_channel(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_offsets: torch.Tensor,
    scaling: float,
    maxpool_k: int = 3,
) -> torch.Tensor:
    """Compute importance scores from Q/K projections (Eq. 2, 15).

    I_j = sum_{i,h} Softmax_j( MaxPool1D_j( S^h_{i,j} ) ),  S^h_{i,j}=q_i^h·k_j^h/√d

    Axis semantics (verified against the official Triton kernel
    _focus_importance_ragged_kernel and SnapKV; see test_focus_importance_axes):
      - i = query index (row), j = key index (column).
      - MaxPool1D and Softmax run along the KEY axis j, once per query row i
        (the standard attention direction).
      - The SUM over queries i (and heads h) is the *column-wise* aggregation
        depicted in Fig. 5 ("Column-Wise Sums of Delta Matrix").
      - The output I_j is therefore indexed by KEY position j (how much a token
        is attended-to by the rest of the block), not by query.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim] (post-RoPE)
        k: Key tensor [total_tokens, num_heads, head_dim] (post-RoPE)
        seq_offsets: CSR-format sequence boundaries [batch_size + 1]
        scaling: Attention scaling factor (1/√d)
        maxpool_k: Maxpool kernel size (default 3)

    Returns:
        importance: [total_tokens] importance score per token (by key position)
    """
    batch_size = len(seq_offsets) - 1
    importance_list = []

    for b in range(batch_size):
        start = seq_offsets[b].item()
        end = seq_offsets[b + 1].item()

        if start >= end:
            continue

        q_b = q[start:end]  # [B, H, d]
        k_b = k[start:end]  # [B, H, d]

        # Intra-block attention scores: S_ij = q_i·k_j/√d → [H, B, B]
        S = torch.einsum("bhd,Bhd->hbB", q_b, k_b) * scaling

        # MaxPool1D along key axis (smoothing)
        H, B, _ = S.shape
        S_flat = S.reshape(H * B, -1).unsqueeze(1)  # [H*B, 1, B]

        # the dimension is padded by maxpool_k // 2 on the two sides so the resulting tensor has the same length as the input tensor
        S_pooled = F.max_pool1d(S_flat, kernel_size=maxpool_k, stride=1, padding=maxpool_k // 2)
        S_pooled = S_pooled.squeeze(1).view(H, B, -1)  # [H, B, B]

        # Softmax over keys (last dimension)
        S_soft = torch.softmax(S_pooled, dim=-1)  # [H, B, B]

        # Sum over query and head dimensions → I_j ∈ R^B
        I = S_soft.sum(dim=(0, 1))  # [B]

        importance_list.append(I)

    if len(importance_list) == 0:
        return torch.empty(0, dtype=q.dtype, device=q.device)

    return torch.cat(importance_list) # [batch_size * block_size] (total_tokens)


def compute_focus_targets(
    mask_lengths: torch.Tensor,
    avg_decoded: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Per-request retention budget — mirrors official ``focus_compute_targets``.

    target = where(len<=0, 0, min(len, max(ceil(max(avg,1)*alpha), 1)))

    IMPORTANT: N_σ is NOT part of the budget here (the earlier
    ``compute_retention_budget`` folded it in, which diverged from the official
    kernel). The paper's Eq. 4 ``max(⌈αN̄⌉, N_σ)`` is realized at *selection*
    time: if at least ``target`` masked tokens exceed mean+std, all of them are
    kept (the N_σ expansion); otherwise the top-``target`` by ΔI are kept. See
    ``select_and_enforce_constraints`` and focus.py:9-30 / 202-213 in
    ~/FOCUS_ORIGIN.

    Args:
        mask_lengths: [batch_size] number of masked (candidate) tokens per request
        avg_decoded: [batch_size] cumulative mean N̄_decoded (≥0; clamped to ≥1)
        alpha: Expansion factor α>1 (default 1.5)

    Returns:
        targets: [batch_size] int32 budget per request
    """
    avg = torch.clamp(avg_decoded.to(torch.float32), min=1.0)
    retain = torch.ceil(avg * alpha)
    retain = torch.clamp(retain, min=1.0)
    lengths_f = mask_lengths.to(torch.float32)
    retain = torch.minimum(lengths_f, retain)
    retain = torch.where(lengths_f <= 0, torch.zeros_like(retain), retain)
    return retain.to(torch.int32)


def compute_should_evict(
    mask_lengths: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """should_evict[b] = (target>0) & (mask_len>target) — official model semantics.

    A request only evicts when its budget is smaller than its masked count;
    otherwise every masked token is retained (no eviction this step).
    """
    return (targets > 0) & (mask_lengths > targets)


def select_and_enforce_constraints(
    delta_I: torch.Tensor,
    mask: torch.Tensor,
    seq_offsets: torch.Tensor,
    targets: torch.Tensor,
    should_evict: torch.Tensor,
    block_length: int,
    block_progress: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Faithful port of ``focus_select_and_enforce_ragged`` (focus.py:142-251).

    Operates on the *processing* set (Phase A: the full block). Among masked
    candidate positions it decides which to retain; non-masked processing
    positions are ALWAYS retained (mirrors ``retain_processing_mask`` init-True
    in the official model code). Steps, per request:

      1. budget select: top-``target`` masked positions by ΔI.
      2. N_σ expansion: threshold = mean+std over masked ΔI; candidates = ΔI≥thr.
         If ``candidate_count >= target`` keep ALL candidates, else keep the
         top-``target`` set from step 1. (This realizes Eq. 4 max(⌈αN̄⌉, N_σ).)
      3. should_evict=False ⇒ keep ALL masked (no eviction this step).
      4. AR-Context: a masked position p is retained if its block-adjacent masked
         successor (p+1, also masked) is retained — i.e. retain the predecessor.
      5. min-keep: if nothing retained, retain all masked (safety net).
      6. Placeholder Integrity: retain masked, not-yet-retained positions that lie
         left of the rightmost retained masked position AND beyond block_progress.

    Args:
        delta_I: [total_tokens] ΔI = I1 − I0 over the processing set.
        mask: [total_tokens] bool, True = masked candidate.
        seq_offsets: [batch_size+1] CSR boundaries over the processing set.
        targets: [batch_size] budget from ``compute_focus_targets``.
        should_evict: [batch_size] bool from ``compute_should_evict``.
        block_length: processing-set length per request (Phase A: block size).
        block_progress: optional [batch_size] rightmost processed position per
            request (FocusState.rightmost_processed). Defaults to −1 (all
            positions treated as unprocessed) when None.

    Returns:
        retain_masks: List[BoolTensor[block_length]] processing-set retain mask
            (non-masked True + selected masked).
        retained_maps: List[LongTensor] sorted retained block indices.
    """
    batch_size = len(seq_offsets) - 1
    device = delta_I.device
    retain_masks: List[torch.Tensor] = []
    retained_maps: List[torch.Tensor] = []

    for b in range(batch_size):
        start = int(seq_offsets[b].item())
        end = int(seq_offsets[b + 1].item())
        target = int(targets[b].item())
        evict = bool(should_evict[b].item())
        progress = -1 if block_progress is None else int(block_progress[b].item())

        retain_mask = torch.zeros(block_length, dtype=torch.bool, device=device)
        if start >= end:
            retain_masks.append(retain_mask)
            retained_maps.append(torch.empty(0, dtype=torch.int64, device=device))
            continue

        mask_b = mask[start:end]
        dI_b = delta_I[start:end]
        # Non-masked processing positions are always retained.
        retain_mask[: (end - start)] = ~mask_b

        mp = torch.where(mask_b)[0]  # masked block positions (sorted ascending)
        if mp.numel() == 0:
            retain_masks.append(retain_mask)
            retained_maps.append(torch.where(retain_mask)[0])
            continue

        dI_m = dI_b[mp]  # ΔI over masked positions, in ascending-position order
        m = mp.numel()

        if not evict:
            # Keep all masked.
            sel = torch.ones(m, dtype=torch.bool, device=device)
        else:
            k = max(min(target, m), 1)
            # Step 1: top-k by ΔI.
            topk_idx = torch.topk(dI_m, k).indices
            sel = torch.zeros(m, dtype=torch.bool, device=device)
            sel[topk_idx] = True
            # Step 2: N_σ expansion (threshold = mean + std over masked ΔI).
            if m == 1:
                cand = torch.ones(m, dtype=torch.bool, device=device)
            else:
                thr = dI_m.mean() + dI_m.std(unbiased=False)
                cand = dI_m >= thr
            if int(cand.sum().item()) >= k:
                sel = cand.clone()

        # Step 4: AR-Context — retain predecessor of a retained adjacent masked
        # successor. One vectorized pass (matches the kernel's single update).
        if m > 1:
            adjacent = (mp[1:] - mp[:-1]) == 1  # masked p,p+1 are block-adjacent
            adjust = adjacent & sel[1:] & (~sel[:-1])
            sel[:-1] = sel[:-1] | adjust

        # Step 5: min-keep safety net.
        if int(sel.sum().item()) == 0:
            sel[:] = True

        # Step 6: Placeholder Integrity — retain masked, not-retained positions
        # left of the rightmost retained masked position and beyond progress.
        retained_positions = mp[sel]
        if retained_positions.numel() > 0:
            rightmost = int(retained_positions.max().item())
            evicted_before = (mp < rightmost) & (~sel)
            is_unprocessed = mp > progress
            need_keep = evicted_before & is_unprocessed
            sel = sel | need_keep

        retain_mask[mp[sel]] = True
        retain_masks.append(retain_mask)
        retained_maps.append(torch.where(retain_mask)[0])

    return retain_masks, retained_maps
