"""Logic tests for FOCUS selection — budget/threshold/AR-context/placeholder.

Pins the official kernel's rules (focus.py:9-30, 142-251):
  - budget has NO N_σ; α→∞ ⇒ target=mask_len ⇒ should_evict=False ⇒ retain all
  - selection = top-target by ΔI, OR all candidates (ΔI≥mean+std) if there are
    at least `target` of them (the N_σ expansion realizing Eq. 4 max(⌈αN̄⌉,N_σ))
  - AR-Context: retain block-adjacent masked predecessor of a retained successor
  - Placeholder Integrity: retain masked, not-retained positions left of the
    rightmost retained masked position (and beyond block_progress)
"""

import torch

from sglang.srt.dllm.algorithm.focus_utils import (
    FocusRuntimeView,
    compute_focus_targets,
    compute_should_evict,
    select_and_enforce_constraints,
)


def _uniform_offsets(batch_size, block_size, device="cpu"):
    return torch.arange(
        0, (batch_size + 1) * block_size, block_size, dtype=torch.int64, device=device
    )


def _mask_lengths(mask, seq_offsets):
    return torch.tensor(
        [
            int(mask[int(seq_offsets[b]):int(seq_offsets[b + 1])].sum().item())
            for b in range(len(seq_offsets) - 1)
        ],
        dtype=torch.int32,
    )


def test_alpha_infinite_no_evict_retains_all():
    """α→∞ ⇒ target=mask_len ⇒ should_evict False ⇒ retain every masked pos."""
    print("Testing α→∞ retains all masked...")
    block_size = 8
    batch_size = 3
    seq_offsets = _uniform_offsets(batch_size, block_size)

    torch.manual_seed(0)
    delta_I = torch.randn(batch_size * block_size)
    mask = torch.zeros(batch_size * block_size, dtype=torch.bool)
    mask[[1, 2, 3, 5]] = True
    mask[[8 + 0, 8 + 4, 8 + 7]] = True
    mask[[16 + 2, 16 + 3, 16 + 4, 16 + 5, 16 + 6]] = True

    mask_lengths = _mask_lengths(mask, seq_offsets)
    avg = torch.tensor([2.0, 1.0, 3.0])
    targets = compute_focus_targets(mask_lengths, avg, alpha=1e9)
    assert torch.equal(targets, mask_lengths.to(targets.dtype)), targets.tolist()
    should_evict = compute_should_evict(mask_lengths, targets)
    assert not should_evict.any(), should_evict.tolist()

    retain_masks, _ = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_size
    )
    for b in range(batch_size):
        assert torch.all(retain_masks[b]), f"req {b} must retain all at α→∞"
    print("  ✓ all positions retained (== LowConfidence anchor)")
    print("α→∞ no-evict: passed! ✓\n")


def test_small_alpha_evicts():
    """Small α with low yield ⇒ target< mask_len ⇒ eviction occurs."""
    print("Testing small α evicts...")
    block_size = 16
    seq_offsets = _uniform_offsets(1, block_size)
    mask = torch.ones(block_size, dtype=torch.bool)
    delta_I = torch.full((block_size,), -1.0)
    delta_I[[3, 4]] = 5.0

    mask_lengths = _mask_lengths(mask, seq_offsets)
    targets = compute_focus_targets(mask_lengths, torch.tensor([1.0]), alpha=1.5)
    assert targets.item() == 2, targets.item()  # ceil(1.5*1)=2
    should_evict = compute_should_evict(mask_lengths, targets)
    assert should_evict.item()

    retain_masks, retained_maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_size
    )
    assert int(retain_masks[0].sum().item()) < block_size, "must evict"
    s = set(retained_maps[0].tolist())
    assert 3 in s and 4 in s, "high-ΔI positions kept"
    print(f"  retained {len(s)}/{block_size}, high-ΔI kept")
    print("small-α eviction: passed! ✓\n")


def test_nsigma_threshold_or_topk():
    """Pin the threshold-OR-topk rule against explicit cases."""
    print("Testing N_σ threshold-OR-topk rule...")
    block_size = 10
    seq_offsets = _uniform_offsets(1, block_size)
    # Non-adjacent masked positions so AR-context adds nothing.
    mask = torch.zeros(block_size, dtype=torch.bool)
    mask[[0, 2, 4, 6, 8]] = True

    # Case A: expansion FIRES. ΔI on masked (ascending pos) = [9,9,-9,-9,-9].
    # mean=-1.8, std≈8.8, thr≈7 ⇒ candidates={pos0,pos2}; target=2 ⇒ count≥target
    # ⇒ selection = candidates. Rightmost retained=2 ⇒ no placeholder add.
    dA = torch.full((block_size,), 0.0)
    dA[[0, 2]] = 9.0
    dA[[4, 6, 8]] = -9.0
    targets = torch.tensor([2], dtype=torch.int32)
    should_evict = torch.tensor([True])
    _, mapsA = select_and_enforce_constraints(
        dA, mask, seq_offsets, targets, should_evict, block_size
    )
    masked_retained_A = sorted(p for p in mapsA[0].tolist() if mask[p])
    assert masked_retained_A == [0, 2], masked_retained_A
    print(f"  ✓ expansion fires: masked retained={masked_retained_A}")

    # Case B: expansion does NOT fire ⇒ top-target. ΔI=[9,8,7,6,5], target=3.
    # thr≈8.41 ⇒ candidates={pos0} (count1<3) ⇒ top-3 = pos {0,2,4}.
    dB = torch.full((block_size,), 0.0)
    dB[0], dB[2], dB[4], dB[6], dB[8] = 9.0, 8.0, 7.0, 6.0, 5.0
    targetsB = torch.tensor([3], dtype=torch.int32)
    _, mapsB = select_and_enforce_constraints(
        dB, mask, seq_offsets, targetsB, should_evict, block_size
    )
    masked_retained_B = sorted(p for p in mapsB[0].tolist() if mask[p])
    assert masked_retained_B == [0, 2, 4], masked_retained_B
    print(f"  ✓ expansion off → top-k: masked retained={masked_retained_B}")
    print("N_σ threshold-OR-topk: passed! ✓\n")


def test_ar_context_predecessor():
    """A retained masked successor pulls in its block-adjacent masked predecessor."""
    print("Testing AR-Context predecessor...")
    block_size = 8
    seq_offsets = _uniform_offsets(1, block_size)
    # Masked at 4,5 (adjacent). Only 5 has high ΔI.
    mask = torch.zeros(block_size, dtype=torch.bool)
    mask[[4, 5]] = True
    delta_I = torch.zeros(block_size)
    delta_I[5] = 10.0
    targets = torch.tensor([1], dtype=torch.int32)
    should_evict = torch.tensor([True])
    _, maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_size
    )
    s = set(maps[0].tolist())
    assert 5 in s, "top retained"
    assert 4 in s, "adjacent masked predecessor retained (AR-context)"
    print(f"  ✓ retained includes predecessor: {sorted(s & {4,5})}")
    print("AR-Context: passed! ✓\n")


def test_placeholder_integrity():
    """Masked, not-retained positions left of rightmost retained are kept."""
    print("Testing Placeholder Integrity...")
    block_size = 8
    seq_offsets = _uniform_offsets(1, block_size)
    # Masked at 1,3,5,7 (non-adjacent → no AR). Select only the last (pos 7).
    mask = torch.zeros(block_size, dtype=torch.bool)
    mask[[1, 3, 5, 7]] = True
    delta_I = torch.zeros(block_size)
    delta_I[7] = 10.0
    targets = torch.tensor([1], dtype=torch.int32)
    should_evict = torch.tensor([True])
    _, maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_size
    )
    s = set(maps[0].tolist())
    # rightmost retained masked = 7 ⇒ masked {1,3,5} (<7) retained as placeholders.
    for p in [1, 3, 5, 7]:
        assert p in s, f"masked {p} ≤ rightmost retained must be kept"
    print(f"  ✓ placeholders kept: {sorted(p for p in s if mask[p])}")
    print("Placeholder Integrity: passed! ✓\n")


def test_placeholder_respects_block_progress():
    """Placeholder only keeps positions beyond block_progress."""
    print("Testing Placeholder block_progress gate...")
    block_size = 8
    seq_offsets = _uniform_offsets(1, block_size)
    mask = torch.zeros(block_size, dtype=torch.bool)
    mask[[1, 3, 5, 7]] = True
    delta_I = torch.zeros(block_size)
    delta_I[7] = 10.0
    targets = torch.tensor([1], dtype=torch.int32)
    should_evict = torch.tensor([True])
    # progress=3 ⇒ only masked positions >3 (i.e. 5) qualify as placeholders; 1,3 excluded.
    _, maps = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, block_size,
        block_progress=torch.tensor([3]),
    )
    masked_kept = sorted(p for p in maps[0].tolist() if mask[p])
    assert masked_kept == [5, 7], masked_kept
    print(f"  ✓ progress=3 ⇒ masked kept={masked_kept} (1,3 gated out)")
    print("Placeholder progress gate: passed! ✓\n")


def test_focus_runtime_view_delta():
    """FocusRuntimeView aggregates per-layer importance into ΔI = I1 − I0."""
    print("Testing FocusRuntimeView ΔI...")
    view = FocusRuntimeView(
        block_size=4, batch_size=1, seq_offsets=torch.tensor([0, 4]),
        importance_layers=(0, 1),
    )
    assert not view.has_importance()
    view.set_layer_importance(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert not view.has_importance()
    view.set_layer_importance(1, torch.tensor([2.0, 2.0, 5.0, 1.0]))
    assert view.has_importance()
    assert torch.allclose(view.get_delta_importance(), torch.tensor([1.0, 0.0, 2.0, -3.0]))
    print("  ✓ ΔI correct")
    print("FocusRuntimeView: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Selection Logic Tests")
    print("=" * 60 + "\n")
    try:
        test_alpha_infinite_no_evict_retains_all()
        test_small_alpha_evicts()
        test_nsigma_threshold_or_topk()
        test_ar_context_predecessor()
        test_placeholder_integrity()
        test_placeholder_respects_block_progress()
        test_focus_runtime_view_delta()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
