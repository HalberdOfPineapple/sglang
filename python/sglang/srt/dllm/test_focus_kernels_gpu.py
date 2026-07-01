"""GPU micro-test: FOCUS Phase-B Triton kernels vs the PyTorch oracles.

Validates that the Triton ports in ``algorithm/focus_kernels.py`` reproduce the
numerical oracles in ``algorithm/focus_utils.py`` (which remain the CPU reference):

  * ``focus_select_and_enforce``  vs ``select_and_enforce_constraints``
    — random ΔI sweeps (exact bool match) + the pinned selection-logic cases
      (α→∞ retain-all, small-α eviction, N_σ threshold-OR-topk, AR-context,
       placeholder integrity, block_progress gate).
  * ``focus_importance``          vs ``compute_importance_side_channel``
    — random q/k (MHA + GQA) with float32 reference (the kernel, like the
      official one, reduces in float32).

Run: python python/sglang/srt/dllm/test_focus_kernels_gpu.py
(skips with a message if no CUDA / triton.)
"""

import torch

from sglang.srt.dllm.algorithm.focus_utils import (
    compute_focus_targets,
    compute_importance_side_channel,
    compute_should_evict,
    select_and_enforce_constraints,
)


def _uniform_offsets(bs, B, device):
    return torch.arange(0, (bs + 1) * B, B, dtype=torch.int64, device=device)


def _oracle_retain_2d(delta_I, mask, seq_offsets, targets, should_evict, B, block_progress):
    retain_masks, _ = select_and_enforce_constraints(
        delta_I, mask, seq_offsets, targets, should_evict, B, block_progress=block_progress
    )
    return torch.stack(retain_masks)


def _check_selection_case(name, delta_I, mask, targets, should_evict, B, block_progress, device):
    from sglang.srt.dllm.algorithm.focus_kernels import focus_select_and_enforce

    bs = targets.numel()
    seq_offsets = _uniform_offsets(bs, B, device)
    ref = _oracle_retain_2d(
        delta_I, mask, seq_offsets, targets, should_evict, B, block_progress
    )
    got = focus_select_and_enforce(
        delta_I, mask, targets, should_evict, B, block_progress=block_progress
    )
    ok = torch.equal(ref, got)
    if not ok:
        for b in range(bs):
            if not torch.equal(ref[b], got[b]):
                print(f"  [{name}] req {b} MISMATCH")
                print(f"    ref={ref[b].int().tolist()}")
                print(f"    got={got[b].int().tolist()}")
    assert ok, f"selection kernel != oracle for case {name}"
    print(f"  ✓ {name}")


def test_selection_random(device):
    print("Testing selection kernel vs oracle (random ΔI sweeps)...")
    torch.manual_seed(0)
    for B in (8, 16, 32, 48):
        for bs in (1, 4, 8):
            for alpha in (1.0, 1.5, 3.0, 1e9):
                delta_I = torch.randn(bs * B, dtype=torch.float32, device=device)
                # random masked pattern (~60% masked), at least 1 masked per req
                mask = torch.rand(bs * B, device=device) < 0.6
                m2d = mask.view(bs, B)
                empty = m2d.sum(dim=1) == 0
                if empty.any():
                    m2d[empty, 0] = True
                mask = m2d.view(-1)
                mask_lengths = mask.view(bs, B).sum(dim=1).to(torch.int32)
                avg = torch.rand(bs, device=device) * 4.0
                targets = compute_focus_targets(mask_lengths, avg, alpha)
                should_evict = compute_should_evict(mask_lengths, targets)
                # exercise both None and explicit block_progress
                for bp in (None, torch.randint(-1, B, (bs,), device=device)):
                    _check_selection_case(
                        f"B{B}_bs{bs}_a{alpha}_bp{'N' if bp is None else 'R'}",
                        delta_I, mask, targets, should_evict, B, bp, device,
                    )
    print("selection random sweep: passed! ✓\n")


def test_selection_pinned_cases(device):
    """Re-run the pinned selection-logic cases through the kernel."""
    print("Testing selection kernel vs oracle (pinned logic cases)...")

    # α→∞ retain-all
    B, bs = 8, 3
    delta_I = torch.randn(bs * B, dtype=torch.float32, device=device)
    mask = torch.zeros(bs * B, dtype=torch.bool, device=device)
    mask[[1, 2, 3, 5]] = True
    mask[[8, 12, 15]] = True
    mask[[18, 19, 20, 21, 22]] = True
    mask_lengths = mask.view(bs, B).sum(dim=1).to(torch.int32)
    avg = torch.tensor([2.0, 1.0, 3.0], device=device)
    targets = compute_focus_targets(mask_lengths, avg, 1e9)
    should_evict = compute_should_evict(mask_lengths, targets)
    _check_selection_case("alpha_inf", delta_I, mask, targets, should_evict, B, None, device)

    # small-α eviction with two high-ΔI positions
    B = 16
    mask = torch.ones(B, dtype=torch.bool, device=device)
    delta_I = torch.full((B,), -1.0, dtype=torch.float32, device=device)
    delta_I[[3, 4]] = 5.0
    targets = torch.tensor([2], dtype=torch.int32, device=device)
    should_evict = torch.tensor([True], device=device)
    _check_selection_case("small_alpha", delta_I, mask, targets, should_evict, B, None, device)

    # N_σ expansion fires
    B = 10
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    mask[[0, 2, 4, 6, 8]] = True
    dA = torch.zeros(B, dtype=torch.float32, device=device)
    dA[[0, 2]] = 9.0
    dA[[4, 6, 8]] = -9.0
    targets = torch.tensor([2], dtype=torch.int32, device=device)
    should_evict = torch.tensor([True], device=device)
    _check_selection_case("nsigma_fires", dA, mask, targets, should_evict, B, None, device)

    # N_σ off → top-target
    dB = torch.zeros(B, dtype=torch.float32, device=device)
    dB[0], dB[2], dB[4], dB[6], dB[8] = 9.0, 8.0, 7.0, 6.0, 5.0
    targets = torch.tensor([3], dtype=torch.int32, device=device)
    _check_selection_case("nsigma_topk", dB, mask, targets, should_evict, B, None, device)

    # AR-context predecessor
    B = 8
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    mask[[4, 5]] = True
    delta_I = torch.zeros(B, dtype=torch.float32, device=device)
    delta_I[5] = 10.0
    targets = torch.tensor([1], dtype=torch.int32, device=device)
    should_evict = torch.tensor([True], device=device)
    _check_selection_case("ar_context", delta_I, mask, targets, should_evict, B, None, device)

    # placeholder integrity + progress gate
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    mask[[1, 3, 5, 7]] = True
    delta_I = torch.zeros(B, dtype=torch.float32, device=device)
    delta_I[7] = 10.0
    targets = torch.tensor([1], dtype=torch.int32, device=device)
    _check_selection_case("placeholder", delta_I, mask, targets, should_evict, B, None, device)
    _check_selection_case(
        "placeholder_progress", delta_I, mask, targets, should_evict, B,
        torch.tensor([3], device=device), device,
    )
    print("selection pinned cases: passed! ✓\n")


def test_importance(device):
    print("Testing importance kernel vs oracle (MHA + GQA)...")
    from sglang.srt.dllm.algorithm.focus_kernels import focus_importance

    torch.manual_seed(1)
    for B in (16, 32):
        for bs in (1, 4):
            for H, Hkv in ((8, 8), (8, 2)):
                d = 64
                total = bs * B
                q = torch.randn(total, H, d, dtype=torch.float32, device=device)
                k_kv = torch.randn(total, Hkv, d, dtype=torch.float32, device=device)
                scale = 1.0 / (d ** 0.5)
                seq_offsets = _uniform_offsets(bs, B, device)

                # Oracle takes k broadcast to H heads.
                if Hkv != H:
                    k_full = k_kv.repeat_interleave(H // Hkv, dim=1)
                else:
                    k_full = k_kv
                ref = compute_importance_side_channel(
                    q, k_full, seq_offsets, scale, maxpool_k=3
                ).to(torch.float32)

                got = focus_importance(
                    q, k_kv, seq_offsets, scale, B,
                    num_key_value_groups=H // Hkv, maxpool_k=3,
                )
                err = (got - ref).abs().max().item()
                assert err < 1e-3, f"importance err {err} (B{B} bs{bs} H{H}/{Hkv})"
                print(f"  ✓ B{B} bs{bs} H{H}/{Hkv}  max_err={err:.2e}")
    print("importance kernel: passed! ✓\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping FOCUS kernel GPU tests.")
        raise SystemExit(0)
    try:
        import triton  # noqa: F401
    except Exception:
        print("triton not available — skipping FOCUS kernel GPU tests.")
        raise SystemExit(0)

    dev = torch.device("cuda")
    print("\n" + "=" * 60)
    print("FOCUS Phase-B Triton Kernel GPU Tests")
    print("=" * 60 + "\n")
    test_selection_random(dev)
    test_selection_pinned_cases(dev)
    test_importance(dev)
    print("=" * 60)
    print("ALL FOCUS KERNEL TESTS PASSED ✓")
    print("=" * 60)
