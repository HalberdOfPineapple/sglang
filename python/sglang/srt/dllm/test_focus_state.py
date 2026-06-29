"""Unit tests for FOCUS state structures (standalone)."""

import torch

from sglang.srt.dllm.mixin.req import FocusState, DelayedCacheState


def test_focus_state():
    """Test FocusState tracking and statistics."""
    print("Testing FocusState...")

    # Test initialization
    state = FocusState(block_length=32)
    assert state.block_length == 32
    assert state.token_sum == 0
    assert state.total_steps == 0
    assert state.rightmost_processed == -1
    print("  ✓ Initialization")

    # Test avg_decoded_tokens with zero steps
    assert state.avg_decoded_tokens == 0.0
    print("  ✓ avg_decoded_tokens with zero steps")

    # Test avg_decoded_tokens computation
    state.token_sum = 15
    state.total_steps = 5
    assert abs(state.avg_decoded_tokens - 3.0) < 1e-6
    print("  ✓ avg_decoded_tokens computation")

    # Test reset_for_new_block
    state.rightmost_processed = 20
    state.reset_for_new_block()
    assert state.token_sum == 15  # Preserved
    assert state.total_steps == 5  # Preserved
    assert state.rightmost_processed == -1  # Reset
    print("  ✓ reset_for_new_block")

    # Test cumulative statistics (simulate second block)
    # Starting from token_sum=15, total_steps=5
    # After reset, add more stats
    old_avg = state.avg_decoded_tokens  # Should be 3.0
    state.token_sum = 23  # Added 8 more tokens total
    state.total_steps = 7  # Added 2 more steps total
    new_avg = state.avg_decoded_tokens  # Should be 23/7
    assert abs(new_avg - 23/7) < 1e-6
    print("  ✓ Cumulative statistics")

    print("FocusState: All tests passed! ✓\n")


def test_delayed_cache_state():
    """Test DelayedCacheState Neighbor-Aware caching."""
    print("Testing DelayedCacheState...")

    # Test initialization
    state = DelayedCacheState(block_length=8, uncached_positions=None)
    assert state.block_length == 8
    assert state.needs_warmup is True
    assert state.uncached_positions.shape == (8,)
    assert state.uncached_positions.all()
    print("  ✓ Initialization")

    # Test get_processing_indices all uncached
    indices = state.get_processing_indices()
    assert torch.equal(indices, torch.arange(8))
    print("  ✓ get_processing_indices (all uncached)")

    # Test get_processing_indices partial
    uncached = torch.tensor([True, False, True, False, True, False, False, False])
    state2 = DelayedCacheState(block_length=8, uncached_positions=uncached)
    indices = state2.get_processing_indices()
    expected = torch.tensor([0, 2, 4])
    assert torch.equal(indices, expected)
    print("  ✓ get_processing_indices (partial)")

    # Test Neighbor-Aware Stability
    state3 = DelayedCacheState(block_length=8, uncached_positions=None)
    mask_id = 0
    dllm_mask = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0])  # 0=masked, positions 2,3,4 decoded
    state3.update_from_mask(dllm_mask, mask_id)
    expected_uncached = torch.tensor([True, True, False, False, True, True, True, True])
    assert torch.equal(state3.uncached_positions, expected_uncached)
    print("  ✓ Neighbor-Aware Stability")

    # Test last position
    state4 = DelayedCacheState(block_length=8, uncached_positions=None)
    dllm_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])  # Only last position decoded
    state4.update_from_mask(dllm_mask, mask_id)
    expected_uncached = torch.tensor([True, True, True, True, True, True, True, False])
    assert torch.equal(state4.uncached_positions, expected_uncached)
    print("  ✓ Last position handling")

    # Test isolated decoded token
    state5 = DelayedCacheState(block_length=8, uncached_positions=None)
    dllm_mask = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])  # Position 3 decoded, neighbors masked
    state5.update_from_mask(dllm_mask, mask_id)
    expected_uncached = torch.tensor([True, True, True, True, True, True, True, True])
    assert torch.equal(state5.uncached_positions, expected_uncached)
    print("  ✓ Isolated decoded token (not cached)")

    # Test incremental updates
    state6 = DelayedCacheState(block_length=8, uncached_positions=None)
    dllm_mask_step1 = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0])
    state6.update_from_mask(dllm_mask_step1, mask_id)
    expected_step1 = torch.tensor([False, True, True, True, True, True, True, True])
    assert torch.equal(state6.uncached_positions, expected_step1)

    dllm_mask_step2 = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    state6.update_from_mask(dllm_mask_step2, mask_id)
    expected_step2 = torch.tensor([False, False, True, True, True, True, True, True])
    assert torch.equal(state6.uncached_positions, expected_step2)
    print("  ✓ Incremental updates")

    # Test reset_for_new_block
    state7 = DelayedCacheState(block_length=8, uncached_positions=None)
    dllm_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    state7.update_from_mask(dllm_mask, mask_id)
    assert not state7.uncached_positions.all()
    assert not state7.needs_warmup

    state7.reset_for_new_block()
    assert state7.uncached_positions.all()
    assert state7.needs_warmup
    print("  ✓ reset_for_new_block")

    print("DelayedCacheState: All tests passed! ✓\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FOCUS State Structures Unit Tests")
    print("="*60 + "\n")

    try:
        test_focus_state()
        test_delayed_cache_state()
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
