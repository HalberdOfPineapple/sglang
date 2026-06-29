"""Logic-level tests for FOCUS selection, including the α→∞ ⇒ retain-all property.

These tests validate the host-side decision logic the Focus algorithm relies on,
without launching a model. The key correctness anchor is: when alpha is large
enough that the budget K saturates to B (the whole block), FOCUS must retain
*every* masked position, i.e. it makes the exact same commit decisions as
LowConfidence (no eviction).
"""

import math

import torch

from sglang.srt.dllm.algorithm.focus_utils import (
    FocusRuntimeView,
    compute_retention_budget,
    select_and_enforce_constraints,
)


def _uniform_offsets(batch_size, block_size, device="cpu"):
    return torch.arange(
        0, (batch_size + 1) * block_size, block_size, dtype=torch.int64, device=device
    )


def test_alpha_infinite_retains_all_masked():
    """α→∞ ⇒ K=B ⇒ every masked position retained (FOCUS == LowConfidence)."""
    print("Testing α→∞ retains all masked positions...")

    block_size = 8
    batch_size = 3
    seq_offsets = _uniform_offsets(batch_size, block_size)

    torch.manual_seed(0)
    delta_I = torch.randn(batch_size * block_size)

    # Arbitrary mask pattern per request.
    mask = torch.zeros(batch_size * block_size, dtype=torch.bool)
    mask[[1, 2, 3, 5]] = True              # req 0
    mask[[8 + 0, 8 + 4, 8 + 7]] = True     # req 1
    mask[[16 + 2, 16 + 3, 16 + 4, 16 + 5, 16 + 6]] = True  # req 2

    avg_decoded = torch.tensor([2.0, 1.0, 3.0])
    huge_alpha = 1e9

    budgets = compute_retention_budget(
        delta_I, avg_decoded, mask, seq_offsets, huge_alpha, block_size
    )
    # Budget saturates at block_size.
    assert torch.all(budgets == block_size), f"budgets={budgets.tolist()}"

    _, retained_maps = select_and_enforce_constraints(
        delta_I, budgets, mask, seq_offsets, block_size, min_retain=1
    )

    for b in range(batch_size):
        start = seq_offsets[b].item()
        end = seq_offsets[b + 1].item()
        masked_positions = torch.where(mask[start:end])[0]
        retained = retained_maps[b]
        # Every masked position must be in the retained set.
        for pos in masked_positions.tolist():
            assert pos in retained.tolist(), (
                f"req {b}: masked pos {pos} not retained "
                f"(retained={sorted(retained.tolist())})"
            )
    print("  ✓ All masked positions retained when K=B")
    print("α→∞ equivalence: passed! ✓\n")


def test_small_alpha_evicts():
    """Small α with low historical yield ⇒ K < #masked ⇒ some eviction happens."""
    print("Testing small α evicts non-decodable positions...")

    block_size = 16
    batch_size = 1
    seq_offsets = _uniform_offsets(batch_size, block_size)

    # All 16 positions masked.
    mask = torch.ones(block_size, dtype=torch.bool)

    # A few positions have clearly higher ΔI (decodable); the rest near-zero.
    delta_I = torch.full((block_size,), -1.0)
    delta_I[[3, 4]] = 5.0  # strongly decodable

    avg_decoded = torch.tensor([1.0])  # low historical yield
    alpha = 1.5

    budgets = compute_retention_budget(
        delta_I, avg_decoded, mask, seq_offsets, alpha, block_size
    )
    K = budgets[0].item()
    # base = ceil(1.5 * 1) = 2; N_sigma counts ΔI ≥ mean+std (the two high ones).
    assert K < block_size, f"expected eviction (K<{block_size}), got K={K}"
    print(f"  Budget K={K} < block_size={block_size}")

    _, retained_maps = select_and_enforce_constraints(
        delta_I, budgets, mask, seq_offsets, block_size, min_retain=1
    )
    retained = retained_maps[0]
    assert len(retained) < block_size, "should evict at least one position"
    # The two strongly-decodable positions must survive.
    assert 3 in retained.tolist() and 4 in retained.tolist()
    print(f"  Retained {len(retained)}/{block_size}; high-ΔI positions kept")
    print("Small-α eviction: passed! ✓\n")


def test_ar_context_preservation():
    """Each retained position's predecessor (i-1) must also be retained."""
    print("Testing AR-Context Preservation...")

    block_size = 8
    batch_size = 1
    seq_offsets = _uniform_offsets(batch_size, block_size)

    mask = torch.ones(block_size, dtype=torch.bool)
    delta_I = torch.zeros(block_size)
    delta_I[5] = 10.0  # only position 5 strongly selected

    budgets = torch.tensor([1], dtype=torch.int32)
    _, retained_maps = select_and_enforce_constraints(
        delta_I, budgets, mask, seq_offsets, block_size, min_retain=1
    )
    retained = set(retained_maps[0].tolist())
    assert 5 in retained, "top position retained"
    assert 4 in retained, "predecessor (AR-context) retained"
    print(f"  Retained: {sorted(retained)} (includes predecessor 4)")
    print("AR-Context Preservation: passed! ✓\n")


def test_placeholder_integrity():
    """All masked positions below max(S) are retained (valid reference KV)."""
    print("Testing Placeholder Integrity...")

    block_size = 8
    batch_size = 1
    seq_offsets = _uniform_offsets(batch_size, block_size)

    # Positions 1,3,5,7 masked; the rest already decoded.
    mask = torch.zeros(block_size, dtype=torch.bool)
    mask[[1, 3, 5, 7]] = True

    delta_I = torch.zeros(block_size)
    delta_I[7] = 10.0  # select the last masked position

    budgets = torch.tensor([1], dtype=torch.int32)
    _, retained_maps = select_and_enforce_constraints(
        delta_I, budgets, mask, seq_offsets, block_size, min_retain=1
    )
    retained = set(retained_maps[0].tolist())
    # max(S) reaches 7, so all masked positions <7 (1,3,5) must be retained.
    for pos in [1, 3, 5, 7]:
        assert pos in retained, f"masked pos {pos} below max(S) must be retained"
    print(f"  Retained: {sorted(retained)} (all masked ≤ max kept)")
    print("Placeholder Integrity: passed! ✓\n")


def test_focus_runtime_view_delta():
    """FocusRuntimeView aggregates per-layer importance into ΔI = I1 - I0."""
    print("Testing FocusRuntimeView ΔI...")

    view = FocusRuntimeView(
        block_size=4,
        batch_size=1,
        seq_offsets=torch.tensor([0, 4]),
        importance_layers=(0, 1),
    )
    assert not view.has_importance()
    view.set_layer_importance(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert not view.has_importance()
    view.set_layer_importance(1, torch.tensor([2.0, 2.0, 5.0, 1.0]))
    assert view.has_importance()
    delta = view.get_delta_importance()
    assert torch.allclose(delta, torch.tensor([1.0, 0.0, 2.0, -3.0]))
    print("  ✓ ΔI computed correctly from two layers")
    print("FocusRuntimeView: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Selection Logic Tests")
    print("=" * 60 + "\n")
    try:
        test_alpha_infinite_retains_all_masked()
        test_small_alpha_evicts()
        test_ar_context_preservation()
        test_placeholder_integrity()
        test_focus_runtime_view_delta()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
