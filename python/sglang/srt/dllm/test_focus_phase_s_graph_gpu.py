"""GPU micro-test: CUDA-graph capture of the FOCUS Phase-S paged-prefill attn.

De-risks the §C keystone WITHOUT the full model: can we capture the exact
FlashInfer paged-prefill call Phase S makes inside a CUDA graph, keyed on a fixed
token **bucket**, and replay it with a smaller real |S| (padded, pad queries
routed to a **scratch** KV slot) so the real rows match an eager reference? This
validates (a) FlashInfer paged-prefill in graph mode (use_cuda_graph=True), (b) the
pad layout + scratch-KV invariant from ``focus_graph.py``, (c) replay with rewritten
ragged indptr. Mirrors ``test_focus_reduced_attention_gpu.py`` (non-causal,
page_size=1) but adds the graph.

Run: LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python <this file>
(skips with a message if no CUDA / flashinfer).
"""

import torch

from sglang.srt.dllm.algorithm.focus_graph import (
    build_phase_s_graph_layout,
    pad_phase_s_tokens,
    phase_s_token_bucket,
)


def _torch_noncausal_attn(q, k, v, scale):
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    a = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", a, v.float())


def test_phase_s_graph_capture_replay():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        return
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except Exception as e:  # pragma: no cover
        print(f"flashinfer unavailable ({e}) — skipping.")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    H, Hkv, d = 8, 8, 128
    scale = d ** -0.5
    C = 10          # per-request context
    max_tokens = 16  # capture bucket cap for this test

    # Paged KV pool, page_size=1. Reserve slot 0 as the scratch slot for pad tokens.
    num_pages = 128
    k_cache = torch.randn(num_pages, 1, Hkv, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_pages, 1, Hkv, d, device=device, dtype=dtype)
    SCRATCH = 0

    # One request: context at slots 1..C, retained block KV at slots C+1..C+S.
    real_S = 3
    bucket = phase_s_token_bucket(real_S, max_tokens)  # = 4
    ctx_slots = torch.arange(1, 1 + C, device=device)
    ret_slots = torch.arange(1 + C, 1 + C + real_S, device=device)
    real_kv_slots = torch.cat([ctx_slots, ret_slots])  # [C+S] context+retained

    q_real = torch.randn(real_S, H, d, device=device, dtype=dtype)

    # --- Eager reference (no graph): q=|S| vs context+retained. ---
    k_ref = k_cache[real_kv_slots, 0]
    v_ref = v_cache[real_kv_slots, 0]
    o_ref = _torch_noncausal_attn(q_real, k_ref, v_ref, scale)

    # --- Build the padded graph layout (bs=1 real + pad segment). ---
    layout = build_phase_s_graph_layout(
        torch.tensor([real_S]), torch.tensor([C]), bucket
    )
    pad_len = layout.pad_len  # bucket - real_S
    # qo_lens=[S, pad], kv_lens=[C+S, pad]; total qo=bucket.
    qo_lens = layout.qo_lens.tolist()
    kv_lens = layout.kv_lens.tolist()

    # Static graph buffers sized to the bucket / max kv.
    nseg = len(qo_lens)  # bs+1
    qo_indptr_buf = torch.zeros(nseg + 1, dtype=torch.int32, device=device)
    kv_indptr_buf = torch.zeros(nseg + 1, dtype=torch.int32, device=device)
    max_kv = max_tokens + C + max_tokens  # generous
    kv_indices_buf = torch.zeros(max_kv, dtype=torch.int32, device=device)
    last_page_buf = torch.ones(nseg, dtype=torch.int32, device=device)
    q_buf = torch.zeros(bucket, H, d, device=device, dtype=dtype)

    def fill_metadata(qo_lens, kv_lens, real_kv_slots, pad_len):
        qo = torch.tensor([0] + list(torch.cumsum(torch.tensor(qo_lens), 0).tolist()),
                          dtype=torch.int32, device=device)
        kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
                          dtype=torch.int32, device=device)
        qo_indptr_buf[: qo.numel()].copy_(qo)
        kv_indptr_buf[: kv.numel()].copy_(kv)
        # kv_indices: real request's context+retained, then pad_len scratch slots.
        pad_slots = torch.full((pad_len,), SCRATCH, dtype=torch.int32, device=device)
        kv_idx = torch.cat([real_kv_slots.to(torch.int32), pad_slots])
        kv_indices_buf[: kv_idx.numel()].copy_(kv_idx)
        return qo, kv, kv_idx.numel()

    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        paged_kv_indptr_buf=kv_indptr_buf,
        paged_kv_indices_buf=kv_indices_buf,
        paged_kv_last_page_len_buf=last_page_buf,
    )

    _, _, kv_total = fill_metadata(qo_lens, kv_lens, real_kv_slots, pad_len)
    # Load q into the static buffer (real rows + zero pad).
    q_buf.copy_(pad_phase_s_tokens(q_real, bucket, 0.0))

    def plan():
        wrapper.plan(
            qo_indptr_buf[: nseg + 1],
            kv_indptr_buf[: nseg + 1],
            kv_indices_buf[:kv_total],
            last_page_buf[:nseg],
            H, Hkv, d, 1,  # page_size
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    plan()

    # --- Capture the graph around wrapper.run(). ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = wrapper.run(q_buf, (k_cache, v_cache))
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o_graph = wrapper.run(q_buf, (k_cache, v_cache))

    plan()  # re-plan into the same buffers (host) before replay
    graph.replay()
    torch.cuda.synchronize()

    o_real = o_graph[:real_S].float()
    err = (o_real - o_ref).abs().max().item()
    assert err < 5e-2, f"graph-replay Phase-S real rows != eager oracle: {err}"
    print(f"  ✓ captured Phase-S graph; real rows match oracle (max err {err:.2e})")

    # --- Replay again with DIFFERENT real content (same bucket) — graph reuse. ---
    q_real2 = torch.randn(real_S, H, d, device=device, dtype=dtype)
    o_ref2 = _torch_noncausal_attn(q_real2, k_ref, v_ref, scale)
    q_buf.copy_(pad_phase_s_tokens(q_real2, bucket, 0.0))
    plan()
    graph.replay()
    torch.cuda.synchronize()
    err2 = (o_graph[:real_S].float() - o_ref2).abs().max().item()
    assert err2 < 5e-2, f"graph reuse with new content mismatch: {err2}"
    print(f"  ✓ graph reused with new q (max err {err2:.2e}) — pad/scratch stable")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOCUS Phase-S CUDA-Graph GPU Micro-Test")
    print("=" * 60 + "\n")
    try:
        test_phase_s_graph_capture_replay()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
