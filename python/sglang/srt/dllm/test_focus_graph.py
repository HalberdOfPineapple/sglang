"""Unit tests for FOCUS §C Phase-S graph foundation (bucketization + pad layout)."""

import torch

from sglang.srt.dllm.algorithm.focus_graph import (
    build_capture_token_buckets,
    build_phase_s_graph_layout,
    pad_phase_s_out_cache_loc,
    pad_phase_s_tokens,
    phase_s_token_bucket,
)


def test_phase_s_token_bucket():
    print("Testing phase_s_token_bucket ladder...")
    M = 512  # max tokens (bs=16, B=32)
    # pow2 up to 256
    assert phase_s_token_bucket(1, M) == 1
    assert phase_s_token_bucket(2, M) == 2
    assert phase_s_token_bucket(3, M) == 4
    assert phase_s_token_bucket(5, M) == 8
    assert phase_s_token_bucket(17, M) == 32
    assert phase_s_token_bucket(200, M) == 256
    assert phase_s_token_bucket(256, M) == 256
    # stride-256 above 256
    assert phase_s_token_bucket(257, M) == 512
    assert phase_s_token_bucket(512, M) == 512
    # clamp at max
    assert phase_s_token_bucket(999, M) == 512
    assert phase_s_token_bucket(0, M) == 1
    print("  ✓ ladder = pow2≤256 then stride-256, clamped")
    print("phase_s_token_bucket: passed! ✓\n")


def test_build_capture_token_buckets():
    print("Testing build_capture_token_buckets...")
    b = build_capture_token_buckets(512)
    assert b == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], b
    b2 = build_capture_token_buckets(768)
    assert b2 == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768], b2
    # every bucket is a valid rung that phase_s_token_bucket can return
    for n in range(1, 513):
        assert phase_s_token_bucket(n, 512) in b
    print(f"  ✓ buckets(512)={b}")
    print("build_capture_token_buckets: passed! ✓\n")


def test_build_phase_s_graph_layout():
    print("Testing build_phase_s_graph_layout (ragged pad segment)...")
    new_lens = torch.tensor([3, 1, 6])  # Σ|S| = 10
    context = torch.tensor([2, 5, 0])
    bucket = phase_s_token_bucket(10, 512)  # = 16
    lay = build_phase_s_graph_layout(new_lens, context, bucket)
    assert lay.bucket == 16
    assert lay.real_tokens == 10
    assert lay.pad_len == 6
    # qo: [3,1,6, pad=6] ; sums to bucket
    assert lay.qo_lens.tolist() == [3, 1, 6, 6]
    assert int(lay.qo_lens.sum()) == bucket
    # kv: [ctx+|S| ..., pad] = [5,6,6, 6]
    assert lay.kv_lens.tolist() == [5, 6, 6, 6]
    assert lay.seg_is_pad.tolist() == [False, False, False, True]
    print(f"  ✓ qo={lay.qo_lens.tolist()} kv={lay.kv_lens.tolist()} pad={lay.pad_len}")
    print("build_phase_s_graph_layout: passed! ✓\n")


def test_layout_no_pad_identity():
    """When Σ|S| == bucket the pad segment is empty (α→∞-style anchor)."""
    print("Testing build_phase_s_graph_layout no-pad identity...")
    new_lens = torch.tensor([8, 8])  # Σ = 16 == bucket(16)
    context = torch.tensor([4, 4])
    lay = build_phase_s_graph_layout(new_lens, context, 16)
    assert lay.pad_len == 0
    assert lay.qo_lens.tolist() == [8, 8, 0]
    assert lay.kv_lens.tolist() == [12, 12, 0]
    assert int(lay.qo_lens.sum()) == 16
    print("  ✓ Σ|S|==bucket ⇒ pad_len=0, real layout preserved")
    print("layout no-pad identity: passed! ✓\n")


def test_pad_out_cache_loc_scratch():
    print("Testing pad_phase_s_out_cache_loc (KV-write safety)...")
    real = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64)  # Σ|S|=5
    scratch = 999
    out = pad_phase_s_out_cache_loc(real, bucket=8, scratch_loc=scratch)
    assert out.tolist() == [10, 11, 12, 20, 21, 999, 999, 999]
    # real prefix preserved, pad → scratch (never a real block slot)
    assert torch.equal(out[:5], real)
    assert (out[5:] == scratch).all()
    print("  ✓ pad rows all point to scratch slot (no real-KV corruption)")
    print("pad_phase_s_out_cache_loc: passed! ✓\n")


def test_pad_tokens():
    print("Testing pad_phase_s_tokens...")
    ids = torch.tensor([5, 6, 7])
    out = pad_phase_s_tokens(ids, bucket=6, pad_value=0)
    assert out.tolist() == [5, 6, 7, 0, 0, 0]
    h = torch.randn(3, 8)
    ph = pad_phase_s_tokens(h, bucket=5, pad_value=0)
    assert ph.shape == (5, 8)
    assert torch.equal(ph[:3], h)
    assert (ph[3:] == 0).all()
    # no-op when already full
    assert torch.equal(pad_phase_s_tokens(ids, 3), ids)
    print("  ✓ per-token tensors padded; real prefix preserved; no-op when full")
    print("pad_phase_s_tokens: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS §C Phase-S Graph Foundation Tests")
    print("=" * 60 + "\n")
    try:
        test_phase_s_token_bucket()
        test_build_capture_token_buckets()
        test_build_phase_s_graph_layout()
        test_layout_no_pad_identity()
        test_pad_out_cache_loc_scratch()
        test_pad_tokens()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
