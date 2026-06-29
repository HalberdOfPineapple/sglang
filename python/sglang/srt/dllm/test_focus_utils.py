"""Unit tests for FOCUS helper functions."""

import torch

from sglang.srt.dllm.algorithm.focus_utils import (
    compute_importance_side_channel,
    compute_retention_budget,
    select_and_enforce_constraints,
)


def test_compute_importance_side_channel():
    """Test importance computation from Q/K."""
    print("Testing compute_importance_side_channel...")

    # Create mock Q/K for 2 sequences: seq0=[4 tokens], seq1=[3 tokens]
    batch_size = 2
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
    assert torch.all(importance >= 0), "Importance should be non-negative (softmax result)"
    assert not torch.any(torch.isnan(importance)), "No NaNs"
    assert not torch.any(torch.isinf(importance)), "No Infs"

    print("  ✓ Shape and validity")

    # Test empty sequence
    q_empty = torch.empty(0, num_heads, head_dim)
    k_empty = torch.empty(0, num_heads, head_dim)
    seq_offsets_empty = torch.tensor([0], dtype=torch.int32)
    importance_empty = compute_importance_side_channel(
        q_empty, k_empty, seq_offsets_empty, scaling, maxpool_k=3
    )
    assert importance_empty.shape == (0,)
    print("  ✓ Empty sequence handling")

    print("compute_importance_side_channel: All tests passed! ✓\n")


def test_compute_retention_budget():
    """Test dynamic budgeting."""
    print("Testing compute_retention_budget...")

    batch_size = 3
    block_length = 8
    alpha = 1.5

    # Sequence offsets: [0, 8, 16, 24]
    seq_offsets = torch.tensor([0, 8, 16, 24], dtype=torch.int32)

    # Delta I: random importance deltas
    torch.manual_seed(42)
    delta_I = torch.randn(24)

    # Masks: all masked for simplicity
    mask = torch.ones(24, dtype=torch.bool)

    # Historical averages: [2.0, 3.0, 4.0]
    avg_decoded = torch.tensor([2.0, 3.0, 4.0])

    budgets = compute_retention_budget(
        delta_I, avg_decoded, mask, seq_offsets, alpha, block_length
    )

    assert budgets.shape == (batch_size,)
    assert torch.all(budgets >= 1), "Budget should be at least 1"
    assert torch.all(budgets <= block_length), "Budget should not exceed block_length"

    # Base budgets: [ceil(1.5*2), ceil(1.5*3), ceil(1.5*4)] = [3, 5, 6]
    # But N_sigma might override
    print(f"  Budgets: {budgets.tolist()}")
    print("  ✓ Shape and constraints")

    # Test with no masked tokens
    mask_none = torch.zeros(24, dtype=torch.bool)
    budgets_none = compute_retention_budget(
        delta_I, avg_decoded, mask_none, seq_offsets, alpha, block_length
    )
    assert torch.all(budgets_none >= 1), "Should return min budget when no masked"
    print("  ✓ No masked tokens handling")

    print("compute_retention_budget: All tests passed! ✓\n")


def test_select_and_enforce_constraints():
    """Test selection with AR-Context and Placeholder Integrity."""
    print("Testing select_and_enforce_constraints...")

    block_length = 8
    batch_size = 2
    seq_offsets = torch.tensor([0, 8, 16], dtype=torch.int32)

    # Seq 0: masked at [2, 3, 4, 5, 6]
    # Seq 1: masked at [1, 3, 5, 7]
    delta_I = torch.zeros(16)
    delta_I[2:7] = torch.tensor([0.1, 0.9, 0.8, 0.3, 0.2])  # Seq 0
    delta_I[8 + 1] = 0.5
    delta_I[8 + 3] = 0.9
    delta_I[8 + 5] = 0.7
    delta_I[8 + 7] = 0.2

    mask = torch.zeros(16, dtype=torch.bool)
    mask[2:7] = True  # Seq 0
    mask[[8 + 1, 8 + 3, 8 + 5, 8 + 7]] = True  # Seq 1

    # Budgets: seq0=2, seq1=2 (will select top-2 by importance)
    budgets = torch.tensor([2, 2], dtype=torch.int32)

    retain_masks, retained_maps = select_and_enforce_constraints(
        delta_I, budgets, mask, seq_offsets, block_length, min_retain=1
    )

    assert len(retain_masks) == batch_size
    assert len(retained_maps) == batch_size

    # Seq 0: Top-2 masked by delta_I are [3, 4] (0.9, 0.8)
    # AR-Context: add predecessors [2, 3] (3-1=2, 4-1=3)
    # So S should include {2, 3, 4}
    # Placeholder Integrity: max(S)=4, so retain all masked < 4: {2, 3}
    # Final S = {2, 3, 4}
    retained_seq0 = retained_maps[0]
    assert 3 in retained_seq0, "Should retain position 3 (top importance)"
    assert 4 in retained_seq0, "Should retain position 4 (top importance)"
    print(f"  Seq 0 retained: {retained_seq0.tolist()}")

    # Seq 1: Top-2 masked are [3, 5] (0.9, 0.7)
    # AR-Context: add [2, 4]
    # Placeholder Integrity: max(S)=5, retain all masked < 5: {1, 3}
    # Final S includes {1, 2, 3, 4, 5} potentially
    retained_seq1 = retained_maps[1]
    assert 3 in retained_seq1, "Should retain position 3 (top importance)"
    print(f"  Seq 1 retained: {retained_seq1.tolist()}")
    print("  ✓ TopK and constraints")

    # Test min_retain
    budgets_zero = torch.zeros(batch_size, dtype=torch.int32)
    retain_masks_min, retained_maps_min = select_and_enforce_constraints(
        delta_I, budgets_zero, mask, seq_offsets, block_length, min_retain=1
    )
    assert all(len(m) >= 1 for m in retained_maps_min), "Should enforce min_retain=1"
    print("  ✓ Minimum retention")

    # Test no masked tokens
    mask_none = torch.zeros(16, dtype=torch.bool)
    retain_masks_none, retained_maps_none = select_and_enforce_constraints(
        delta_I, budgets, mask_none, seq_offsets, block_length, min_retain=1
    )
    assert all(len(m) == 0 for m in retained_maps_none), "No retention when no masked"
    print("  ✓ No masked tokens handling")

    print("select_and_enforce_constraints: All tests passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Helper Functions Unit Tests")
    print("=" * 60 + "\n")

    try:
        test_compute_importance_side_channel()
        test_compute_retention_budget()
        test_select_and_enforce_constraints()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
