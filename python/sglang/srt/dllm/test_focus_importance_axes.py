"""Pin the axis semantics of compute_importance_side_channel against Eq. 2.

The FOCUS importance (Eq. 2) is
    I_j = sum_{i,h} Softmax_j( MaxPool1D_j( S^h_{i,j} ) ),   S^h_{i,j} = q_i^h . k_j^h / sqrt(d)
i.e. for each (query i, head h): MaxPool over the KEY axis, Softmax over the KEY
axis (the standard attention direction), then SUM over the QUERY axis i (and
heads h). The importance index j is the KEY index. This matches the official
Triton kernel (_focus_importance_ragged_kernel) which launches one program per
(query, head), softmaxes over keys, and atomic-adds the per-key weights.

These tests compare the vectorized implementation to an explicit, unambiguous
double-loop reference, and assert the directionality with hand-built inputs.
"""

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.focus_utils import compute_importance_side_channel


def _reference_importance(q_b, k_b, scaling, maxpool_k):
    """Literal Eq. 2 for a single block. q_b,k_b: [B, H, d]. Returns [B] over keys."""
    B, H, d = q_b.shape
    I = torch.zeros(B, dtype=torch.float32)
    pad = maxpool_k // 2
    for h in range(H):
        for i in range(B):  # query
            # raw scores over keys j for this (query i, head h)
            s = torch.empty(B, dtype=torch.float32)
            for j in range(B):  # key
                s[j] = (q_b[i, h] @ k_b[j, h]) * scaling
            # MaxPool1D over the KEY axis, -inf padding (matches kernel boundary)
            s_pad = F.pad(s.view(1, 1, B), (pad, pad), value=float("-inf"))
            s_pooled = F.max_pool1d(s_pad, kernel_size=maxpool_k, stride=1).view(B)
            # Softmax over the KEY axis
            w = torch.softmax(s_pooled, dim=-1)
            # accumulate into the KEY position j, summed over queries i and heads h
            I += w
    return I


def test_matches_eq2_reference():
    print("Testing vectorized impl == explicit Eq.2 double loop...")
    torch.manual_seed(0)
    B, H, d = 6, 3, 8
    q = torch.randn(B, H, d)
    k = torch.randn(B, H, d)
    scaling = 1.0 / (d ** 0.5)
    seq_offsets = torch.tensor([0, B], dtype=torch.int64)

    got = compute_importance_side_channel(q, k, seq_offsets, scaling, maxpool_k=3)
    ref = _reference_importance(q, k, scaling, maxpool_k=3)

    assert got.shape == (B,)
    assert torch.allclose(got, ref, atol=1e-5), (
        f"\n got={got}\n ref={ref}\n diff={(got-ref).abs().max()}"
    )
    print(f"  ✓ max|got-ref| = {(got-ref).abs().max().item():.2e}")
    # Per-query softmax sums to 1 over keys, so total importance == H*B (each of
    # the B queries contributes mass 1 per head).
    assert abs(got.sum().item() - H * B) < 1e-3
    print(f"  ✓ sum(I) == H*B == {H*B}")
    print("Eq.2 reference match: passed! ✓\n")


def test_importance_indexes_keys_not_queries():
    """A key attended to by everyone must score high; a lone query must not."""
    print("Testing importance is indexed by KEY (attended-to), not query...")
    B, H, d = 5, 1, 4
    # Make key 2 a "hub": its key vector aligns with all query vectors.
    q = torch.zeros(B, H, d)
    k = torch.zeros(B, H, d)
    q[:, 0, 0] = 1.0           # every query points along axis 0
    k[2, 0, 0] = 10.0          # only key 2 has large component along axis 0
    # other keys point elsewhere (orthogonal) so they get little attention
    for j in range(B):
        if j != 2:
            k[j, 0, 1] = 1.0
    seq_offsets = torch.tensor([0, B], dtype=torch.int64)

    I = compute_importance_side_channel(q, k, seq_offsets, 1.0, maxpool_k=1)
    # Key 2 (and, with maxpool spreading, its neighbours) should dominate.
    assert I.argmax().item() == 2, f"expected key 2 to dominate, got {I}"
    print(f"  ✓ argmax importance = key {I.argmax().item()} (the hub)")
    print("Key-indexing directionality: passed! ✓\n")


def test_maxpool_uses_neg_inf_padding():
    """Confirm F.max_pool1d pads with -inf (boundary keys not polluted by 0)."""
    print("Testing maxpool boundary padding semantics...")
    # All-negative scores: zero padding would corrupt the boundary max.
    x = torch.tensor([[-5.0, -3.0, -4.0, -9.0]]).unsqueeze(0)  # [1,1,4]
    pooled = F.max_pool1d(x, kernel_size=3, stride=1, padding=1).view(-1)
    # position 0: max(pad, -5, -3) must be -3 (not 0 from zero-padding)
    assert pooled[0].item() == -3.0, f"pos0={pooled[0].item()} (zero-padding bug)"
    assert pooled[3].item() == -4.0, f"pos3={pooled[3].item()}"
    print(f"  ✓ pooled={pooled.tolist()} (neg-inf padding confirmed)")
    print("MaxPool padding: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Importance Axis-Semantics Tests")
    print("=" * 60 + "\n")
    try:
        test_maxpool_uses_neg_inf_padding()
        test_matches_eq2_reference()
        test_importance_indexes_keys_not_queries()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
