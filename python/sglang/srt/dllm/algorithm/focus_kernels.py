"""FOCUS Triton kernels (Phase B) — importance scoring + selection/enforcement.

These are the GPU ports of the two loop-bearing host helpers that dominate the
eager per-step host cost (F3-lite: ``select ~14%`` + the per-request importance
einsum). Each kernel is a faithful re-implementation of the corresponding PyTorch
oracle in ``focus_utils.py`` (which stays as the numerical reference / CPU test
oracle) and mirrors the official FOCUS kernels in
``~/FOCUS_ORIGIN/lmdeploy/pytorch/kernels/cuda/focus.py``:

  * ``focus_select_and_enforce``  ⇔ oracle ``select_and_enforce_constraints``
        official ``_focus_select_enforce_ragged_kernel``
  * ``focus_importance``          ⇔ oracle ``compute_importance_side_channel``
        official ``_focus_importance_ragged_kernel``

Design note vs the official ragged kernels: the SGLang FOCUS integration always
runs the *prefix* selection/importance over a **uniform ``block_size`` processing
set per request** (Phase P is the full block), so we do NOT need the official
CSR ``INDPTR`` ragged machinery — grid is simply ``(batch_size,)`` (selection) /
``(batch_size, H, block_size)`` (importance) with a contiguous ``block_size`` slice
per request. This keeps the kernels much simpler than the LMDeploy originals while
computing the identical result. If a ragged (delayed-cache-shrunk) processing set
is added later, port the INDPTR variant.

All statistics (mean/std for the N_σ threshold, softmax) are computed in float32,
matching the *official* kernel (the bf16 torch oracle was a lossy approximation;
the float32 result is the true reference and is what these kernels reproduce).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# §B2 — selection + constraint enforcement
# ---------------------------------------------------------------------------
@triton.jit
def _focus_select_enforce_kernel(
    DELTA_I,          # [bs*B] float  — ΔI over the (uniform) processing set
    MASK,             # [bs*B] bool   — True at masked candidate positions
    TARGETS,          # [bs] int32    — per-request budget (compute_focus_targets)
    SHOULD,           # [bs] int8     — per-request should_evict
    BLOCK_PROGRESS,   # [bs] int32    — rightmost processed position (or -1)
    OUT,              # [bs*B] bool   — retain mask (written)
    block_size,       # runtime int   — B (uniform processing length per request)
    BLOCK: tl.constexpr,  # next_pow2(B)
):
    seq = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_bounds = offs < block_size
    base = seq * block_size

    scores_row = tl.load(DELTA_I + base + offs, mask=in_bounds, other=0.0).to(tl.float32)
    masked = tl.load(MASK + base + offs, mask=in_bounds, other=0).to(tl.int1)
    valid_row = masked & in_bounds
    valid_f32 = valid_row.to(tl.float32)
    counts = tl.sum(valid_f32, axis=0)

    # Budget clamp (matches compute_should_evict + the official kernel head):
    # not-evict ⇒ target 0 ⇒ empty top-k / no threshold ⇒ all masked retained.
    target = tl.load(TARGETS + seq).to(tl.int32)
    should_evict = tl.load(SHOULD + seq).to(tl.int1)
    target = tl.where(should_evict, target, 0)
    target = tl.maximum(target, 0)
    max_counts = counts.to(tl.int32)
    target = tl.minimum(target, max_counts)
    positive = target > 0
    target_clamped = tl.where(positive, tl.maximum(target, 1), target)

    # Step 1: iterative top-``target`` masked positions by ΔI (reproduces
    # torch.topk for distinct scores; picks lowest lane on ties). scores_rank is
    # a scratch copy so the ORIGINAL scores_row survives for the mean/std below.
    selected = tl.zeros([BLOCK], dtype=tl.int1)
    remaining = target_clamped
    filler = float("-inf")
    scores_rank = scores_row
    for _ in range(0, BLOCK):
        available = valid_row & (~selected)
        available_count = tl.sum(available.to(tl.int32), axis=0)
        work = (available_count > 0) & (remaining > 0)
        masked_scores = tl.where(available, scores_rank, filler)
        best_val = tl.max(masked_scores, axis=0)
        select_mask = available & (masked_scores == best_val)
        prefix = tl.cumsum(select_mask.to(tl.int32))
        take = select_mask & (prefix <= remaining) & (prefix > 0)
        take = tl.where(work, take, tl.zeros_like(take))
        selected = selected | take
        taken = tl.sum(take.to(tl.int32), axis=0)
        remaining = remaining - taken
        scores_rank = tl.where(take, filler, scores_rank)

    # Step 2: N_σ expansion — threshold = mean + std (unbiased=False) over the
    # masked ΔI. If at least ``target`` masked positions clear it, keep them all;
    # else fall back to the top-``target`` set (Eq. 4: max(⌈αN̄⌉, N_σ)).
    masked_scores = scores_row * valid_f32
    denom = tl.maximum(counts, 1.0)
    mean = tl.sum(masked_scores, axis=0) / denom
    diff = (scores_row - mean) * valid_f32
    variance = tl.sum(diff * diff, axis=0) / denom
    std = tl.sqrt(variance)
    threshold = mean + std
    candidate_mask = (scores_row >= threshold) & valid_row
    candidate_counts = tl.sum(candidate_mask.to(tl.int32), axis=0)
    use_threshold = (target_clamped > 0) & (candidate_counts >= target_clamped)
    selection = tl.where(use_threshold, candidate_mask, selected)
    selection = selection & valid_row

    # not-evict ⇒ keep ALL masked (selection is ignored).
    sel = tl.where(should_evict, selection, valid_row)
    sel = sel & valid_row

    # Publish base_retain (non-masked processing positions are always kept) so the
    # AR-context pass can read the retained state of the block-adjacent successor.
    base_retain = (masked == 0) & in_bounds
    base_retain = base_retain | sel
    tl.store(OUT + base + offs, base_retain, mask=in_bounds)
    tl.debug_barrier()  # order the store before the shifted reload below

    # Step 4: AR-Context — a masked position i whose block-adjacent masked
    # successor i+1 is retained is itself retained (retain the predecessor).
    has_next = (offs + 1) < block_size
    next_masked = tl.load(MASK + base + offs + 1, mask=has_next, other=0).to(tl.int1)
    next_retain = tl.load(OUT + base + offs + 1, mask=has_next, other=0).to(tl.int1)
    adjacency = masked & next_masked & has_next
    adjust = adjacency & next_retain & (sel == 0)
    sel = sel | adjust

    # Step 5: min-keep safety net — if nothing survived, retain all masked.
    keep_count = tl.sum((sel & valid_row).to(tl.int32), axis=0)
    no_keep = keep_count == 0
    sel = tl.where(no_keep, valid_row, sel)

    # Step 6: Placeholder Integrity — retain masked, not-retained positions that
    # lie left of the rightmost retained masked position AND beyond block_progress.
    retain_valid = sel & valid_row
    safe_positions = tl.where(retain_valid, offs, -1)
    rightmost = tl.max(safe_positions, axis=0)
    less_than_rightmost = offs < rightmost
    evicted_before = less_than_rightmost & (sel == 0) & valid_row
    progress = tl.load(BLOCK_PROGRESS + seq).to(tl.int32)
    is_unprocessed = offs > progress
    need_keep = is_unprocessed & evicted_before
    sel = sel | need_keep

    retain_full = ((masked == 0) & in_bounds) | (sel & valid_row)
    tl.store(OUT + base + offs, retain_full, mask=in_bounds)


def _pick_focus_select_enforce_meta(width: int):
    """num_warps/num_stages heuristic (ported from the official kernel)."""
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        if width >= 256:
            return 8, 3
        if width >= 128:
            return 4, 3
        if width >= 64:
            return 4, 2
        return 2, 2
    if major >= 8:
        if width >= 256:
            return 4, 2
        if width >= 128:
            return 2, 2
        return 1, 2
    return 1, 1


def focus_select_and_enforce(
    delta_I: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    should_evict: torch.Tensor,
    block_size: int,
    block_progress: torch.Tensor = None,
) -> torch.Tensor:
    """Triton port of ``select_and_enforce_constraints`` for a uniform block set.

    Args:
        delta_I: [bs*block_size] ΔI over the processing set (request-major).
        mask: [bs*block_size] bool, True at masked candidate positions.
        targets: [bs] int budget from ``compute_focus_targets``.
        should_evict: [bs] bool from ``compute_should_evict``.
        block_size: uniform processing length per request (== self.block_size).
        block_progress: optional [bs] rightmost processed position; -1 when None.

    Returns:
        retain_mask_2d: [bs, block_size] bool retain mask (non-masked processing
        positions + selected masked positions), matching ``_select_retained``.
    """
    bs = int(targets.numel())
    device = delta_I.device
    out = torch.empty(bs * block_size, dtype=torch.bool, device=device)
    if block_progress is None:
        block_progress = torch.full((bs,), -1, dtype=torch.int32, device=device)
    else:
        block_progress = block_progress.to(torch.int32).contiguous()

    BLOCK = triton.next_power_of_2(block_size)
    num_warps, num_stages = _pick_focus_select_enforce_meta(block_size)
    grid = (bs,)
    _focus_select_enforce_kernel[grid](
        delta_I.contiguous().to(torch.float32),
        mask.contiguous(),
        targets.contiguous().to(torch.int32),
        should_evict.contiguous().to(torch.int8),
        block_progress,
        out,
        block_size,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.view(bs, block_size)


# ---------------------------------------------------------------------------
# §B1 — importance scoring (ragged over a uniform block set)
# ---------------------------------------------------------------------------
@triton.jit
def _focus_importance_kernel(
    Q,               # [total_tokens, H,   d]
    K,               # [total_tokens, Hkv, d]
    SEQ_OFFSETS,     # [bs+1] int  — CSR boundaries over the processing set
    WORKSPACE,       # [grid_rows, BLOCK_ROW] float32 scratch (per (seq,head,query))
    IMPORTANCE,      # [total_tokens] float32 accumulator (atomic_add target)
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_ws,
    max_seq,         # max processing length (== block_size for uniform)
    head_dim,
    scale,
    rows_per_seq,    # H * max_seq
    BLOCK_D: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    num_key_value_groups: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // rows_per_seq
    head_row = pid % rows_per_seq
    head_idx = head_row // max_seq
    query_idx = head_row % max_seq

    seq_start = tl.load(SEQ_OFFSETS + seq_idx).to(tl.int64)
    seq_end = tl.load(SEQ_OFFSETS + seq_idx + 1).to(tl.int64)
    seq_len = seq_end - seq_start
    if (seq_len <= 0) | (query_idx >= seq_len):
        return

    kv_head_idx = head_idx // num_key_value_groups
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # Uniform processing set: token index = seq_start + local position.
    q_token = seq_start + query_idx
    q_ptr = Q + q_token * stride_qt + head_idx * stride_qh
    q_vec = tl.load(q_ptr + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)

    neg_inf = -float("inf")
    row_ws = WORKSPACE + pid * stride_ws

    # 3-point (k=3) MaxPool over keys with -inf padding, streamed as prev/curr/next.
    cond = 0 < seq_len
    key_token = seq_start + 0
    k_ptr = K + key_token * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(tl.float32)
    curr_score = tl.where(cond, tl.sum(q_vec * k_vec, axis=0) * scale, neg_inf)

    cond = 1 < seq_len
    key_token = seq_start + 1
    k_ptr = K + key_token * stride_kt + kv_head_idx * stride_kh
    k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond, other=0.0).to(tl.float32)
    next_score = tl.where(cond, tl.sum(q_vec * k_vec, axis=0) * scale, neg_inf)

    prev_score = neg_inf
    for key_pos in tl.range(0, max_seq):
        key_valid = key_pos < seq_len
        next_valid = (key_pos + 1) < seq_len
        pooled = tl.maximum(prev_score, curr_score)
        pooled = tl.maximum(pooled, tl.where(next_valid, next_score, neg_inf))
        tl.store(row_ws + key_pos, tl.where(key_valid, pooled, neg_inf))
        prev_score = tl.where(key_valid, curr_score, prev_score)
        curr_score = tl.where(next_valid, next_score, curr_score)
        future = key_pos + 2
        cond_f = future < seq_len
        key_token = seq_start + future
        k_ptr = K + key_token * stride_kt + kv_head_idx * stride_kh
        k_vec = tl.load(k_ptr + offs_d * stride_kd, mask=mask_d & cond_f, other=0.0).to(tl.float32)
        next_score = tl.where(cond_f, tl.sum(q_vec * k_vec, axis=0) * scale, neg_inf)

    # Softmax over keys (two passes over the padded workspace; -inf ⇒ 0 weight).
    offs_block = tl.arange(0, BLOCK_ROW)
    row_max = neg_inf
    for start in tl.range(0, max_seq, BLOCK_ROW):
        blk = start + offs_block
        m = blk < max_seq
        vals = tl.load(row_ws + blk, mask=m, other=neg_inf)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    row_sum = tl.zeros([1], dtype=tl.float32)
    for start in tl.range(0, max_seq, BLOCK_ROW):
        blk = start + offs_block
        m = blk < max_seq
        vals = tl.load(row_ws + blk, mask=m, other=neg_inf)
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)
    row_sum = tl.where(row_sum > 0, row_sum, 1.0)
    inv = 1.0 / row_sum

    # Column-wise accumulation: add softmax weight for key j into importance[j],
    # summed over all queries i and heads h (atomic — many pids target j).
    imp_row = IMPORTANCE + seq_start
    for start in tl.range(0, max_seq, BLOCK_ROW):
        blk = start + offs_block
        m = blk < seq_len
        vals = tl.load(row_ws + blk, mask=m, other=neg_inf)
        weights = tl.exp(vals - row_max) * inv
        tl.atomic_add(imp_row + blk, weights, mask=m)


def _pick_focus_importance_meta(block_d: int, block_row: int):
    """num_warps/num_stages heuristic (ported from the official kernel)."""
    major, _ = torch.cuda.get_device_capability()
    area = block_d * block_row
    if block_row <= 32 and block_d <= 128:
        return (2, 2) if major >= 8 else (1, 1)
    if major >= 9:
        if area >= 16384:
            return 8, 4
        if area >= 8192:
            return 8, 3
        if area >= 4096:
            return 4, 3
        return 2, 2
    if major >= 8:
        if area >= 16384:
            return 4, 3
        if area >= 8192:
            return 4, 2
        return 2, 2
    return 2, 2


def focus_importance(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    seq_offsets: torch.Tensor,
    scale: float,
    max_seq: int,
    num_key_value_groups: int = 1,
    maxpool_k: int = 3,
) -> torch.Tensor:
    """Triton port of ``compute_importance_side_channel`` (uniform block set).

    I_j = Σ_{i,h} Softmax_j( MaxPool1D_{k=3}( q_i^h · k_j^h · scale ) ), indexed by
    KEY position j. Reproduces the oracle in float32.

    Args:
        query_states: [total_tokens, H, d] post-RoPE queries.
        key_states: [total_tokens, Hkv, d] post-RoPE keys (GQA supported via
            ``num_key_value_groups = H // Hkv``; pass broadcast k + groups=1 to
            match the current model call).
        seq_offsets: [bs+1] CSR boundaries over the processing set (uniform ⇒
            arange(0, (bs+1)*block, block)).
        scale: attention scale (1/√d).
        max_seq: max processing length (== block_size for the uniform prefix).
        num_key_value_groups: H // Hkv.
        maxpool_k: only k=3 is implemented (the paper/official default).

    Returns:
        importance: [total_tokens] float32 (key-indexed).
    """
    if maxpool_k != 3:
        raise NotImplementedError(
            f"focus_importance Triton kernel only supports maxpool_k=3, got {maxpool_k}"
        )
    if max_seq <= 0:
        return query_states.new_zeros((0,), dtype=torch.float32)
    device = query_states.device
    num_seq = int(seq_offsets.numel() - 1)
    num_heads = query_states.size(1)
    head_dim = query_states.size(2)
    total_tokens = query_states.size(0)
    rows_per_seq = num_heads * max_seq
    total_rows = rows_per_seq * num_seq

    importance = torch.zeros((total_tokens,), dtype=torch.float32, device=device)
    block_d = triton.next_power_of_2(head_dim)
    block_row = triton.next_power_of_2(max(16, min(max_seq, 128)))
    # Workspace holds one pooled score per key (max_seq wide); BLOCK_ROW only
    # tiles the softmax reduction over it.
    workspace = torch.empty((total_rows, max_seq), dtype=torch.float32, device=device)

    stride_qt, stride_qh, stride_qd = query_states.stride()
    stride_kt, stride_kh, stride_kd = key_states.stride()
    num_warps, num_stages = _pick_focus_importance_meta(block_d, block_row)
    grid = (total_rows,)
    _focus_importance_kernel[grid](
        query_states,
        key_states,
        seq_offsets,
        workspace,
        importance,
        stride_qt, stride_qh, stride_qd,
        stride_kt, stride_kh, stride_kd,
        workspace.stride(0),
        max_seq,
        head_dim,
        scale,
        rows_per_seq,
        BLOCK_D=block_d,
        BLOCK_ROW=block_row,
        num_key_value_groups=num_key_value_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return importance
