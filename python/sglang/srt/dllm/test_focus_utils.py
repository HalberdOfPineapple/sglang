"""Unit tests for FOCUS helper functions (budget/selection mirror official kernels)."""

import torch

from sglang.srt.dllm.algorithm.focus_utils import (
    compute_focus_targets,
    compute_importance_side_channel,
    compute_should_evict,
    select_and_enforce_constraints,
)


def _uniform_offsets(batch_size, block_size, device="cpu"):
    return torch.arange(
        0, (batch_size + 1) * block_size, block_size, dtype=torch.int64, device=device
    )


def _masked_lengths(mask, seq_offsets):
    return torch.tensor(
        [
            int(mask[int(seq_offsets[b]):int(seq_offsets[b + 1])].sum().item())
            for b in range(len(seq_offsets) - 1)
        ],
        dtype=torch.int32,
    )


def test_compute_importance_side_channel():
    """Test importance computation from Q/K."""
    print("Testing compute_importance_side_channel...")

    num_heads = 2
    head_dim = 4
    seq_lens = [4, 3]
    total_tokens = sum(seq_lens)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, head_dim)
    k = torch.randn(total_tokens, num_heads, head_dim)
    seq_offsets = torch.tensor([0, 4, 7], dtype=torch.int32)
    scaling = 1.0 / (head_dim ** 0.5)

    importance = compute_importance_side_channel(q, k, seq_offsets, scaling, maxpool_k=3)

    assert importance.shape == (total_tokens,)
    assert torch.all(importance >= 0)
    assert not torch.any(torch.isnan(importance))
    assert not torch.any(torch.isinf(importance))
    print("  ✓ Shape and validity")

    q_empty = torch.empty(0, num_heads, head_dim)
    k_empty = torch.empty(0, num_heads, head_dim)
    seq_offsets_empty = torch.tensor([0], dtype=torch.int32)
    importance_empty = compute_importance_side_channel(
        q_empty, k_empty, seq_offsets_empty, scaling, maxpool_k=3
    )
    assert importance_empty.shape == (0,)
    print("  ✓ Empty sequence handling")
    print("compute_importance_side_channel: All tests passed! ✓\n")


def test_compute_focus_targets():
    """Budget = min(len, max(ceil(max(avg,1)*alpha),1)); NO N_sigma."""
    print("Testing compute_focus_targets...")
    alpha = 1.5

    # avg=[2,3,4] -> ceil(1.5*avg)=[3,5,6], clamped to mask_len.
    mask_lengths = torch.tensor([8, 8, 8], dtype=torch.int32)
    avg = torch.tensor([2.0, 3.0, 4.0])
    targets = compute_focus_targets(mask_lengths, avg, alpha)
    assert targets.tolist() == [3, 5, 6], targets.tolist()
    print(f"  ✓ targets={targets.tolist()} (no N_sigma in budget)")

    # avg<1 clamps to 1 -> ceil(1.5)=2.
    targets2 = compute_focus_targets(
        torch.tensor([8], dtype=torch.int32), torch.tensor([0.0]), alpha
    )
    assert targets2.item() == 2, targets2.item()
    print("  ✓ avg clamped to >=1")

    # mask_len=0 -> target 0; small mask_len clamps the budget.
    targets3 = compute_focus_targets(
        torch.tensor([0, 2], dtype=torch.int32), torch.tensor([5.0, 5.0]), alpha
    )
    assert targets3.tolist() == [0, 2], targets3.tolist()
    print("  ✓ zero-length and length-clamped budgets")
    print("compute_focus_targets: All tests passed! ✓\n")


def test_compute_should_evict():
    """should_evict = (target>0) & (mask_len>target)."""
    print("Testing compute_should_evict...")
    mask_lengths = torch.tensor([8, 3, 0, 5], dtype=torch.int32)
    targets = torch.tensor([3, 3, 0, 5], dtype=torch.int32)
    se = compute_should_evict(mask_lengths, targets)
    # 8>3 -> True; 3>3 -> False; target0 -> False; 5>5 -> False.
    assert se.tolist() == [True, False, False, False], se.tolist()
    print(f"  ✓ should_evict={se.tolist()}")
    print("compute_should_evict: All tests passed! ✓\n")


def test_select_basic_topk_and_constraints():
    """Selection retains top-target + AR-context predecessor + non-masked."""
    print("Testing select_and_enforce_constraints (top-k path)...")

    block_length = 8
    seq_offsets = _uniform_offsets(2, block_length)

    delta_I = torch.zeros(16)
    # Seq 0 masked at [2,3,4,5,6]; high dI at 3,4.
    delta_I[2:7] = torch.tensor([0.1, 0.9, 0.8, 0.3, 0.2])
    # Seq 1 masked at [1,3,5,7]; high dI at 3,5.
    delta_I[8 + 1] = 0.5
    delta_I[8 + 3] = 0.9
    delta_I[8 + 5] = 0.7
    delta_I[8 + 7] = 0.2

    mask = torch.zeros(16, dtype=torch.bool)
    mask[2:7] = True
    mask[[8 + 1, 8 + 3, 8 + 5, 8 + 7]] = True

    mask_lengths = _masked_lengths(mask, seq_offsets)  # [5, 4]
    # Force budget=2 so we exercise the top-k path (avg tuned so ceil(1.5*avg)=2).
    targets = torch.tensor([2, 2], dtype=torch.int32)
    should_evict = compute_should_evict(mask_lengths, targets)  # [True, True]

    retain_masks, retained_maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_length
    )

    # Non-masked positions must always be retained.
    for b in range(2):
        rm = retain_masks[b]
        mb = mask[b * block_length:(b + 1) * block_length]
        assert torch.all(rm[~mb]), "all non-masked retained"

    s0 = set(retained_maps[0].tolist())
    # Top-2 masked dI in seq0 are pos 3,4; AR-context adds predecessor 2 (adjacent
    # masked, since 3's pred 2 is masked & 4's pred 3 retained). Placeholder keeps
    # masked < max retained. Non-masked 0,1,7 retained.
    assert 3 in s0 and 4 in s0
    assert 2 in s0, "AR-context predecessor"
    print(f"  Seq0 retained: {sorted(s0)}")

    s1 = set(retained_maps[1].tolist())
    assert 3 in s1 and 5 in s1
    print(f"  Seq1 retained: {sorted(s1)}")
    print("  ✓ top-k + AR-context + non-masked retention")
    print("select_and_enforce (top-k): passed! ✓\n")


def test_select_no_evict_retains_all_masked():
    """should_evict=False ⇒ every masked position retained."""
    print("Testing select_and_enforce_constraints (no-evict path)...")
    block_length = 8
    seq_offsets = _uniform_offsets(1, block_length)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[[1, 2, 4, 6]] = True
    delta_I = torch.randn(8)
    targets = torch.tensor([4], dtype=torch.int32)  # == mask_len -> no evict
    should_evict = compute_should_evict(_masked_lengths(mask, seq_offsets), targets)
    assert should_evict.item() is False or should_evict.item() == False  # noqa: E712

    retain_masks, retained_maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_length
    )
    # All positions retained (masked + non-masked).
    assert torch.all(retain_masks[0]), "no-evict retains everything"
    print("  ✓ no-evict retains all")
    print("select_and_enforce (no-evict): passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Helper Functions Unit Tests")
    print("=" * 60 + "\n")
    try:
        test_compute_importance_side_channel()
        test_compute_focus_targets()
        test_compute_should_evict()
        test_select_basic_topk_and_constraints()
        test_select_no_evict_retains_all_masked()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
