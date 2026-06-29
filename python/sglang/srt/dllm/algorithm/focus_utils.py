"""FOCUS helper functions for importance, budgeting, and selection.

Phase A: PyTorch reference implementations.
Phase B: Will be replaced with Triton kernels.
"""

import math
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


def compute_retention_budget(
    delta_I: torch.Tensor,
    avg_decoded: torch.Tensor,
    mask: torch.Tensor,
    seq_offsets: torch.Tensor,
    alpha: float,
    block_length: int,
) -> torch.Tensor:
    """Compute dynamic retention budget K per request (Eq. 4, 5).

    Args:
        delta_I: [total_tokens] importance delta (I1 - I0)
        avg_decoded: [batch_size] cumulative mean N̄_decoded
        mask: [total_tokens] boolean mask (True = masked/candidate)
        seq_offsets: [batch_size + 1] CSR sequence boundaries
        alpha: Expansion factor (default 1.5)
        block_length: Block size

    Returns:
        budgets: [batch_size] retention budget K per request
    """
    batch_size = len(seq_offsets) - 1
    budgets = []

    for b in range(batch_size):
        start = seq_offsets[b].item()
        end = seq_offsets[b + 1].item()

        if start >= end:
            budgets.append(1)  # Min retention
            continue

        delta_I_b = delta_I[start:end]
        mask_b = mask[start:end]

        # Only consider masked positions
        delta_I_masked = delta_I_b[mask_b]

        if len(delta_I_masked) == 0:
            budgets.append(1)
            continue

        # Statistical threshold: μ + σ (1σ above mean). With a single masked
        # token std is undefined; treat that token as above-threshold (N_σ=1).
        if len(delta_I_masked) == 1:
            N_sigma = 1
        else:
            threshold = delta_I_masked.mean() + delta_I_masked.std()
            N_sigma = (delta_I_masked >= threshold).sum().item()

        # Base budget from historical average
        base_budget = math.ceil(alpha * avg_decoded[b].item())

        # Final budget: K = min(B, max(⌈α·N̄⌉, N_σ))
        K = min(block_length, max(base_budget, N_sigma))
        budgets.append(K)

    return torch.tensor(budgets, dtype=torch.int32, device=delta_I.device)


def select_and_enforce_constraints(
    delta_I: torch.Tensor,
    budgets: torch.Tensor,
    mask: torch.Tensor,
    seq_offsets: torch.Tensor,
    block_length: int,
    min_retain: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Select tokens with structural constraints (Algorithm 1).

    Constraints:
    1. TopK selection from masked positions
    2. AR-Context Preservation: retain predecessor i-1 for each selected i
    3. Placeholder Integrity: retain all masked j < max(S)
    4. Minimum retention: |S| >= min_retain

    Args:
        delta_I: [total_tokens] importance delta
        budgets: [batch_size] retention budget K per request
        mask: [total_tokens] boolean mask (True = masked/candidate)
        seq_offsets: [batch_size + 1] CSR sequence boundaries
        block_length: Block size
        min_retain: Minimum tokens to retain (default 1)

    Returns:
        retain_masks: List[Tensor[block_length]] boolean masks per request
        retained_maps: List[Tensor[variable]] retained indices per request
    """
    batch_size = len(seq_offsets) - 1
    retain_masks = []
    retained_maps = []

    for b in range(batch_size):
        start = seq_offsets[b].item()
        end = seq_offsets[b + 1].item()
        K = budgets[b].item()

        if start >= end:
            # Empty sequence
            retain_mask = torch.zeros(block_length, dtype=torch.bool, device=delta_I.device)
            retained_map = torch.empty(0, dtype=torch.int64, device=delta_I.device)
            retain_masks.append(retain_mask)
            retained_maps.append(retained_map)
            continue

        delta_I_b = delta_I[start:end]
        mask_b = mask[start:end]

        # TopK among masked positions
        masked_indices = torch.where(mask_b)[0]

        if len(masked_indices) == 0:
            # No masked tokens
            retain_mask = torch.zeros(block_length, dtype=torch.bool, device=delta_I.device)
            retained_map = torch.empty(0, dtype=torch.int64, device=delta_I.device)
            retain_masks.append(retain_mask)
            retained_maps.append(retained_map)
            continue

        delta_I_masked = delta_I_b[masked_indices]
        K_actual = min(K, len(masked_indices))
        topk_in_masked = torch.topk(delta_I_masked, K_actual).indices
        S = masked_indices[topk_in_masked]

        # AR-Context Preservation: add i-1 for each i in S
        if len(S) > 0:
            predecessors = S - 1
            predecessors = predecessors[predecessors >= 0]
            S = torch.unique(torch.cat([S, predecessors]))

        # Placeholder Integrity: retain all masked j < max(S)
        if len(S) > 0:
            max_S = S.max().item()
            placeholder_indices = torch.arange(max_S + 1, device=delta_I.device)
            # Only add if they're masked
            placeholder_indices = placeholder_indices[mask_b[:max_S + 1]]
            S = torch.unique(torch.cat([S, placeholder_indices]))

        # Minimum retention
        if len(S) < min_retain:
            # Add top importance masked tokens to reach min_retain
            remaining = min_retain - len(S)
            candidates = masked_indices[~torch.isin(masked_indices, S)]
            if len(candidates) > 0:
                remaining_topk = torch.topk(
                    delta_I_b[candidates], min(remaining, len(candidates))
                ).indices
                S = torch.cat([S, candidates[remaining_topk]])
                S = torch.unique(S)

        # Build retain_mask
        retain_mask = torch.zeros(block_length, dtype=torch.bool, device=delta_I.device)
        retain_mask[S] = True

        retain_masks.append(retain_mask)
        retained_maps.append(S)

    return retain_masks, retained_maps
