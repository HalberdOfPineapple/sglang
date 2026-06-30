"""GPU micro-test: FOCUS reduced-attention mechanism vs a torch dense oracle.

De-risks the keystone of the split forward (plan §8, risks R4/R1) on real
hardware, WITHOUT the full model: it exercises the exact FlashInfer paged-prefill
call SGLang's dLLM-extend path makes (non-causal / ENCODER_ONLY, page_size=1,
token-granular kv_indices) and checks two claims:

  Phase A1: q=|S| retained queries against the FULL block KV (context+B) returns
            EXACTLY the retained rows of the full-block forward. This is the
            paper-exact L1-attention regime and confirms FlashInfer accepts
            q_len(|S|) < kv_len(context+B) (R4).
  Phase S : q=|S| queries against context + the RETAINED block KV (context+|S|),
            with retained KV gathered to a contiguous page prefix, matches the
            torch dense oracle restricted to those KV. Confirms the
            contiguous-prefix compaction reads exactly context+retained (R1/R3).

Run: python python/sglang/srt/dllm/test_focus_reduced_attention_gpu.py
(skips with a message if no CUDA / flashinfer.)
"""

import torch


def _torch_noncausal_attn(q, k, v, scale):
    """Dense non-causal attention oracle. q:[Sq,H,d] k,v:[Sk,H,d] -> [Sq,H,d]."""
    # scores: [H, Sq, Sk]
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    a = torch.softmax(s, dim=-1)
    o = torch.einsum("hqk,khd->qhd", a, v.float())
    return o


def _run_paged(wrapper, q, k_cache, v_cache, kv_indices, qo_len, kv_len, scale, H, Hkv, d):
    """Mirror SGLang's dLLM paged-prefill call: page_size=1, causal=False."""
    device = q.device
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(1, dtype=torch.int32, device=device)
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices.to(torch.int32),
        kv_last_page_len,
        H,
        Hkv,
        d,
        1,  # page_size
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    o = wrapper.forward(
        q.view(-1, H, d),
        (k_cache, v_cache),
        causal=False,
        sm_scale=scale,
    )
    return o.view(qo_len, H, d)


def test_reduced_attention_mechanism():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU micro-test.")
        return
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except Exception as e:  # pragma: no cover
        print(f"flashinfer unavailable ({e}) — skipping.")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    H, Hkv, d = 8, 8, 128  # head config; MHA for a clean oracle
    scale = d ** -0.5
    C = 10   # context length (already-decoded prefix)
    B = 8    # block size
    total = C + B

    # Paged KV pool, page_size=1: one page per token slot. Layout [num_pages,1,Hkv,d].
    num_pages = 64
    k_cache = torch.randn(num_pages, 1, Hkv, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_pages, 1, Hkv, d, device=device, dtype=dtype)

    # Slots 0..C-1 hold context KV; slots C..C+B-1 hold the full block KV (written
    # in Phase P). Block queries are B fresh vectors.
    q_block = torch.randn(B, H, d, device=device, dtype=dtype)
    full_kv_slots = torch.arange(0, total, device=device)  # context + full block

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # --- Reference: full-block forward (q=B vs context+B). ---
    k_ctx_blk = k_cache[full_kv_slots, 0]  # [total, Hkv, d]
    v_ctx_blk = v_cache[full_kv_slots, 0]
    o_full_fi = _run_paged(
        wrapper, q_block, k_cache, v_cache, full_kv_slots, B, total, scale, H, Hkv, d
    )
    o_full_oracle = _torch_noncausal_attn(q_block, k_ctx_blk, v_ctx_blk, scale)
    err_full = (o_full_fi.float() - o_full_oracle).abs().max().item()
    assert err_full < 5e-2, f"full-block FI vs oracle mismatch: {err_full}"
    print(f"  ✓ full-block forward matches oracle (max err {err_full:.2e})")

    # Retained subset S of the B block positions (e.g. evict ~5/8).
    retained = torch.tensor([0, 2, 5], device=device)  # |S| = 3
    S = retained.numel()
    q_ret = q_block[retained]

    # --- Phase A1: q=|S| vs FULL block KV (context+B). ---
    o_a1 = _run_paged(
        wrapper, q_ret, k_cache, v_cache, full_kv_slots, S, total, scale, H, Hkv, d
    )
    # Must equal the retained rows of the full-block forward.
    err_a1 = (o_a1.float() - o_full_oracle[retained]).abs().max().item()
    assert err_a1 < 5e-2, f"Phase A1 reduced != full retained rows: {err_a1}"
    print(f"  ✓ Phase A1 (q=|S|<kv) reproduces retained rows (max err {err_a1:.2e})")

    # --- Phase S: q=|S| vs context + RETAINED block, gathered to a page prefix. ---
    # Retained block KV physically lives at the block's first |S| slots (C..C+S-1).
    # Write the retained block K/V there (block-prefix compaction).
    ret_block_slots = torch.arange(C, C + S, device=device)
    # Build "retained K/V" — for the oracle, just reuse the retained block slots'
    # KV (in the real model these are freshly-computed L>=2 KV; here we only test
    # the addressing/compaction reads context+retained, so any content works).
    retained_kv = torch.randn(S, 1, Hkv, d, device=device, dtype=dtype)
    k_cache[ret_block_slots] = retained_kv
    v_cache[ret_block_slots] = torch.randn(S, 1, Hkv, d, device=device, dtype=dtype)
    # kv_indices = contiguous slice [0 .. C+S-1] (context + block prefix).
    phase_s_slots = torch.arange(0, C + S, device=device)
    o_s = _run_paged(
        wrapper, q_ret, k_cache, v_cache, phase_s_slots, S, C + S, scale, H, Hkv, d
    )
    k_s = k_cache[phase_s_slots, 0]
    v_s = v_cache[phase_s_slots, 0]
    o_s_oracle = _torch_noncausal_attn(q_ret, k_s, v_s, scale)
    err_s = (o_s.float() - o_s_oracle).abs().max().item()
    assert err_s < 5e-2, f"Phase S reduced vs oracle mismatch: {err_s}"
    print(f"  ✓ Phase S (context+|S| prefix) matches oracle (max err {err_s:.2e})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Reduced-Attention GPU Micro-Test")
    print("=" * 60 + "\n")
    try:
        test_reduced_attention_mechanism()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
