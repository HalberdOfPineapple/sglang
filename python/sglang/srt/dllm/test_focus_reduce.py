"""Unit tests for FOCUS reduced-forward compaction helpers (vs torch oracle)."""

import torch

from sglang.srt.dllm.algorithm.focus_reduce import (
    build_retained_index,
    cu_seqlens_from_lens,
    focus_compact_states,
)


def _ref_keep_index(retained_maps, seq_offsets):
    """Independent double-loop reference for the global gather index."""
    idx = []
    lens = []
    for b, rm in enumerate(retained_maps):
        base = int(seq_offsets[b].item())
        n = 0
        for p in rm.tolist():
            idx.append(base + p)
            n += 1
        lens.append(n)
    return idx, lens


def test_build_retained_index():
    print("Testing build_retained_index...")
    block = 8
    seq_offsets = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
    retained_maps = [
        torch.tensor([0, 1, 3, 7]),
        torch.tensor([2, 4]),
        torch.tensor([0, 1, 2, 3, 4, 5]),
    ]
    keep_index, new_lens = build_retained_index(retained_maps, seq_offsets)

    ref_idx, ref_lens = _ref_keep_index(retained_maps, seq_offsets)
    assert keep_index.tolist() == ref_idx, (keep_index.tolist(), ref_idx)
    assert new_lens.tolist() == ref_lens, (new_lens.tolist(), ref_lens)
    # Request-major, ascending within request; global indices in range.
    assert keep_index.tolist() == [0, 1, 3, 7, 10, 12, 16, 17, 18, 19, 20, 21]
    assert new_lens.tolist() == [4, 2, 6]
    print(f"  ✓ keep_index={keep_index.tolist()}")
    print(f"  ✓ new_lens={new_lens.tolist()}")
    print("build_retained_index: passed! ✓\n")


def test_focus_compact_states_matches_oracle():
    print("Testing focus_compact_states vs index_select oracle...")
    total_tokens = 24
    hidden = 16
    num_heads, head_dim = 4, 8

    torch.manual_seed(0)
    h = torch.randn(total_tokens, hidden)
    res = torch.randn(total_tokens, hidden)
    q = torch.randn(total_tokens, num_heads, head_dim)
    pos = torch.arange(total_tokens, dtype=torch.int64)

    seq_offsets = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
    retained_maps = [
        torch.tensor([0, 1, 3, 7]),
        torch.tensor([2, 4]),
        torch.tensor([0, 1, 2, 3, 4, 5]),
    ]
    keep_index, new_lens = build_retained_index(retained_maps, seq_offsets)

    out = focus_compact_states(
        keep_index,
        {"hidden": h, "residual": res, "query": q, "positions": pos, "none": None},
    )

    assert torch.equal(out["hidden"], h[keep_index])
    assert torch.equal(out["residual"], res[keep_index])
    assert torch.equal(out["query"], q[keep_index])
    assert torch.equal(out["positions"], pos[keep_index])
    assert out["none"] is None
    assert out["hidden"].shape == (int(new_lens.sum()), hidden)
    assert out["query"].shape == (int(new_lens.sum()), num_heads, head_dim)
    print(f"  ✓ compacted {total_tokens} → {int(new_lens.sum())} tokens, all tensors match")
    print("focus_compact_states: passed! ✓\n")


def test_cu_seqlens_from_lens():
    print("Testing cu_seqlens_from_lens...")
    lens = torch.tensor([4, 2, 6], dtype=torch.int32)
    cu = cu_seqlens_from_lens(lens)
    assert cu.tolist() == [0, 4, 6, 12], cu.tolist()
    # Empty / zero-length requests handled.
    cu2 = cu_seqlens_from_lens(torch.tensor([0, 3, 0], dtype=torch.int32))
    assert cu2.tolist() == [0, 0, 3, 3], cu2.tolist()
    print(f"  ✓ cu_seqlens={cu.tolist()}")
    print("cu_seqlens_from_lens: passed! ✓\n")


def test_alpha_inf_identity():
    """When every position is retained, compaction is the identity (anchor)."""
    print("Testing α→∞ compaction identity...")
    total_tokens = 16
    h = torch.randn(total_tokens, 8)
    seq_offsets = torch.tensor([0, 8, 16], dtype=torch.int64)
    retained_maps = [torch.arange(8), torch.arange(8)]
    keep_index, new_lens = build_retained_index(retained_maps, seq_offsets)
    out = focus_compact_states(keep_index, {"h": h})
    assert torch.equal(out["h"], h), "retain-all ⇒ identity"
    assert new_lens.tolist() == [8, 8]
    print("  ✓ retain-all ⇒ identity gather")
    print("α→∞ identity: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Reduce (compaction) Tests")
    print("=" * 60 + "\n")
    try:
        test_build_retained_index()
        test_focus_compact_states_matches_oracle()
        test_cu_seqlens_from_lens()
        test_alpha_inf_identity()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
