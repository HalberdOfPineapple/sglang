"""Unit tests for the FOCUS split-forward metadata builder (focus_forward.py).

Pure tensor math — no model / CUDA needed. Validates the per-phase seq_lens /
extend_prefix_lens / extend_seq_lens arithmetic (the keystone of plan §8), the
contiguous-prefix out_cache_loc gather, and the α→∞ identity anchor (|S|=B ⇒ the
reduced phases reproduce the original full-block dLLM-extend metadata).
"""

from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.focus_forward import (
    PHASE_A1,
    PHASE_S,
    build_phase_s_out_cache_loc,
    compute_focus_phase_lens,
    make_focus_phase_batch,
)


def _orig_dllm_extend_lens(seq_lens, block_size):
    """Reference: the original full-block dLLM-extend metadata."""
    context = seq_lens - block_size
    extend_prefix = context.clone()
    extend_seq = torch.full_like(seq_lens, block_size)
    return seq_lens.clone(), extend_prefix, extend_seq


def test_phase_a1_lens():
    print("Testing Phase A1 lens (q=|S|, kv=context+B)...")
    block_size = 8
    # context_b = [10, 4]; seq_lens = context + B
    seq_lens = torch.tensor([18, 12], dtype=torch.int64)
    new_lens = torch.tensor([3, 5], dtype=torch.int64)
    sl, ep, es = compute_focus_phase_lens(PHASE_A1, seq_lens, block_size, new_lens)
    # KV unchanged (full block still cached for A1).
    assert sl.tolist() == [18, 12], sl.tolist()
    # query count = |S|.
    assert es.tolist() == [3, 5], es.tolist()
    # prefix = seq_lens - |S|  ⇒ qo_indptr count = |S|.
    assert ep.tolist() == [15, 7], ep.tolist()
    # By construction query count (seq_lens - prefix) == |S|.
    assert (sl - ep).tolist() == new_lens.tolist()
    print(f"  ✓ seq_lens={sl.tolist()} prefix={ep.tolist()} q={es.tolist()}")
    print("Phase A1 lens: passed! ✓\n")


def test_phase_s_lens():
    print("Testing Phase S lens (q=|S|, kv=context+|S|)...")
    block_size = 8
    seq_lens = torch.tensor([18, 12], dtype=torch.int64)
    new_lens = torch.tensor([3, 5], dtype=torch.int64)
    sl, ep, es = compute_focus_phase_lens(PHASE_S, seq_lens, block_size, new_lens)
    # context = [10, 4]; kv = context + |S|.
    assert sl.tolist() == [13, 9], sl.tolist()
    assert es.tolist() == [3, 5], es.tolist()
    # prefix = context (= original seq_lens - B).
    assert ep.tolist() == [10, 4], ep.tolist()
    assert (sl - ep).tolist() == new_lens.tolist()
    print(f"  ✓ seq_lens={sl.tolist()} prefix={ep.tolist()} q={es.tolist()}")
    print("Phase S lens: passed! ✓\n")


def test_alpha_inf_identity_lens():
    """|S|=B ⇒ both reduced phases reproduce the full-block dLLM-extend metadata."""
    print("Testing α→∞ metadata identity (|S|=B)...")
    block_size = 8
    seq_lens = torch.tensor([18, 12, 8], dtype=torch.int64)
    new_lens = torch.full_like(seq_lens, block_size)  # retain all
    ref_sl, ref_ep, ref_es = _orig_dllm_extend_lens(seq_lens, block_size)
    for phase in (PHASE_A1, PHASE_S):
        sl, ep, es = compute_focus_phase_lens(phase, seq_lens, block_size, new_lens)
        assert torch.equal(sl, ref_sl), (phase, sl.tolist())
        assert torch.equal(ep, ref_ep), (phase, ep.tolist())
        assert torch.equal(es, ref_es), (phase, es.tolist())
    print("  ✓ both phases == original full-block extend metadata")
    print("α→∞ metadata identity: passed! ✓\n")


def test_phase_s_out_cache_loc():
    print("Testing Phase S out_cache_loc (block-prefix gather)...")
    block_size = 4
    # 2 requests, block-contiguous slots.
    out_cache_loc = torch.tensor([100, 101, 102, 103, 200, 201, 202, 203])
    new_lens = torch.tensor([2, 3])
    sel = build_phase_s_out_cache_loc(out_cache_loc, block_size, new_lens)
    # first 2 of req0's block, first 3 of req1's block.
    assert sel.tolist() == [100, 101, 200, 201, 202], sel.tolist()
    # α→∞: retain all ⇒ identity.
    sel_full = build_phase_s_out_cache_loc(
        out_cache_loc, block_size, torch.tensor([4, 4])
    )
    assert torch.equal(sel_full, out_cache_loc), "retain-all ⇒ identity slots"
    print(f"  ✓ retained slots={sel.tolist()}")
    print("Phase S out_cache_loc: passed! ✓\n")


def test_make_focus_phase_batch():
    print("Testing make_focus_phase_batch (field stamping)...")
    block_size = 4
    base = SimpleNamespace(
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        seq_lens=torch.tensor([6, 6], dtype=torch.int64),  # context=2 each, B=4
        seq_lens_cpu=torch.tensor([6, 6]),
        seq_lens_sum=12,
        out_cache_loc=torch.arange(8),
        extend_seq_lens=torch.tensor([4, 4]),
        extend_prefix_lens=torch.tensor([2, 2]),
        focus_view="SHOULD_BE_CLEARED",
        req_pool_indices=torch.tensor([0, 1]),
    )
    new_lens = torch.tensor([1, 2])
    compact_ids = torch.tensor([0, 4, 5])  # |S| = 1 + 2 = 3
    compact_pos = torch.tensor([2, 2, 3])

    # Phase S: KV compacted, out_cache_loc supplied.
    s_loc = build_phase_s_out_cache_loc(base.out_cache_loc, block_size, new_lens)
    fb = make_focus_phase_batch(
        base, PHASE_S, block_size, new_lens, compact_ids, compact_pos, s_loc
    )
    assert fb.seq_lens.tolist() == [3, 4], fb.seq_lens.tolist()  # context+|S|
    assert fb.extend_prefix_lens.tolist() == [2, 2]
    assert fb.extend_seq_lens.tolist() == [1, 2]
    assert fb.extend_num_tokens == 3
    assert fb.seq_lens_sum == 7
    assert fb.out_cache_loc.tolist() == [0, 4, 5]
    assert torch.equal(fb.input_ids, compact_ids)
    assert torch.equal(fb.positions, compact_pos)
    assert fb.focus_view is None, "focus_view must be cleared in the suffix"
    # base untouched (shallow copy).
    assert base.focus_view == "SHOULD_BE_CLEARED"
    assert base.seq_lens.tolist() == [6, 6]

    # Phase A1: KV unchanged, no out_cache_loc override.
    fb1 = make_focus_phase_batch(
        base, PHASE_A1, block_size, new_lens, compact_ids, compact_pos, None
    )
    assert fb1.seq_lens.tolist() == [6, 6], fb1.seq_lens.tolist()  # unchanged
    assert fb1.extend_prefix_lens.tolist() == [5, 4]  # seq_lens - |S|
    assert fb1.extend_seq_lens.tolist() == [1, 2]
    assert torch.equal(fb1.out_cache_loc, base.out_cache_loc)  # not overridden
    print("  ✓ Phase S + A1 fields stamped; base untouched")
    print("make_focus_phase_batch: passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Split-Forward Metadata Builder Tests")
    print("=" * 60 + "\n")
    try:
        test_phase_a1_lens()
        test_phase_s_lens()
        test_alpha_inf_identity_lens()
        test_phase_s_out_cache_loc()
        test_make_focus_phase_batch()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
