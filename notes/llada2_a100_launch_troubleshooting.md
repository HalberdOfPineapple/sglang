# LLaDA2.0-mini on 4x A100 80GB — Launch Troubleshooting

Record of bringing up `inclusionAI/LLaDA2.0-mini` (dLLM, `--dllm-algorithm LowConfidence`) with `--tp-size 4 --ep-size 4` on this proxied A100 cluster. Four distinct failures, each with root cause and fix.

## Model facts
`model_type=llada2_moe`, diffusion LM. ~17B total params (256 experts, top-8, `moe_intermediate_size=512`, 20 layers, hidden 2048, vocab 157184), ~1.4B active. bf16 weights ~34 GB → ~8.5 GB/GPU under TP=4/EP=4. `meta_info.num_retractions` in responses = diffusion remasking count (tokens un-decoded and re-masked during iterative denoising).

## 1. HF download stalls at 0 B/s (Xet vs proxy)
Symptom: `hf download` and SGLang auto-download hang at `Downloading (incomplete total...) 0.00/30.0G [..., ?B/s]`. Server never passes "Load weight begin", so curl to the not-yet-up HTTP server hangs.
Cause: `huggingface_hub` 1.19 + `hf-xet` 1.5.1 default to the Xet CAS endpoint, unreachable through the cluster proxy (`HTTP(S)_PROXY=127.0.0.1:7890`). The Rust Xet client doesn't honor the proxy env vars.
Fix: `export HF_HUB_DISABLE_XET=1` → classic HTTPS downloader via the `huggingface.co` resolve CDN (proxy handles it). `HF_HUB_ENABLE_HF_TRANSFER` is deprecated/ignored here. Don't run two downloads of the same repo concurrently — they deadlock on `.locks/`.

## 2. FlashInfer topk crash on A100 (sm80)
Symptom: at CUDA graph capture, `flashinfer.utils.BackendSupportedError: fused_topk_deepseek does not support compute capability 80`.
Cause: `biased_grouped_topk_gpu` in `python/sglang/srt/layers/moe/topk.py` gated the FlashInfer `fused_topk_deepseek` path on kernel *importability* only, not GPU SM. FlashInfer imports on A100; the kernel needs sm90+ and rejects sm80 at call time.
Fix (local patch on `main`): import `get_device_sm`, add `_device_sm = get_device_sm() if _is_cuda else 0`, add `and _device_sm >= 90` to the gate. A100 falls through to the existing `moe_fused_gate` sgl_kernel (same grouped-topk semantics).

## 3. OOM during CUDA graph capture (sparse MoE + high mem-fraction)
Symptom: OOM in `init_device_graphs` despite weights being only ~8.5 GB/GPU on 80 GB cards.
Cause: default `--mem-fraction-static` (~0.9) auto-sizes the token pool to fill the GPU, leaving ~8 GB free; the 256-expert MoE's graph-capture/dispatch workspaces exceed that.
Fix: `--mem-fraction-static 0.7` → ~21.7 GB free/GPU, capture uses ~1 GB and succeeds. General rule: sparse MoE has tiny per-GPU weights but large capture workspaces; leave headroom.

## 4. curl hangs / returns empty against the local server
Symptom: server is "ready", but `curl http://127.0.0.1:30000/...` returns nothing and no prefill log line appears.
Cause: shell proxy env routes the loopback request into the proxy (`no_proxy` unset), dropping it.
Fix: `curl --noproxy '*' http://127.0.0.1:30000/...` or `export NO_PROXY=localhost,127.0.0.1`.

## Working invocation
```bash
HF_HUB_DISABLE_XET=1 python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini --dllm-algorithm LowConfidence \
  --host 0.0.0.0 --port 30000 --trust-remote-code \
  --tp-size 4 --ep-size 4 --mem-fraction-static 0.7

curl --noproxy '*' http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"inclusionAI/LLaDA2.0-mini","messages":[{"role":"user","content":"..."}],"max_tokens":64}'
```
Verified: `/generate` returns coherent text, e2e ~1 s for 32 tokens, `max_total_num_tokens≈4.97M`, ~21.7 GB free/GPU after capture.

## Open items
- The topk.py SM guard is an uncommitted local change — re-apply / upstream if the tree is updated.
- Consider whether `--mem-fraction-static` can go higher (toward 0.8) now that capture headroom is understood, if a larger KV/token pool is wanted.
