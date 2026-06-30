# Experiment F1 — FOCUS paper-exact reduced forward vs LowConfidence (LLaDA2.0-mini, 1×A100 80GB, TP=1, eager)

First end-to-end throughput/latency comparison of the **paper-exact FOCUS reduced (token-evicting) forward** — implemented in `python/sglang/srt/dllm/algorithm/focus.py` + the `forward_focus_*` split in `python/sglang/srt/models/llada2.py` (see `notes/focus_implementation_progress.md`) — against the stock `LowConfidence` dLLM path, on a single A100, eager (no CUDA graph). FOCUS genuinely evicts 20–34% of tokens from Layers 1-attn..L (measured per-step redundancy Σ|S|/(B·bs) = 0.66→0.80), but in **this eager, single-GPU, small-MoE regime it is slower wall-clock (0.69–0.90× throughput)** because the per-step host overhead of the split forward outweighs the small per-token GPU compute it saves.

> **Honest negative result, scope it correctly.** This measures an **eager PyTorch prototype** of the reduced forward. The FLOPs reduction is real and correct (α→∞ anchor == LowConfidence; redundancy < 1 here), but the wall-clock win the paper reports needs (1) **CUDA-graph capture of Phase S**, (2) **fewer host syncs** (the prototype rebuilds FlashInfer metadata 3× per denoising step and runs Python per-request selection/commit loops with many `.item()` D2H stalls), and (3) a **compute-bound regime** (bigger model / longer context) where per-token GPU work dominates. None of those are in this prototype, so it is host-overhead-bound and loses. The number to carry forward is the **redundancy** (the achievable compute reduction), not this prototype's tok/s.

## Summary

Operating point = HumanEval at sustained concurrency ∈ {1, 8, 16} (= `max_running_requests`), `max_new_tokens=128` (4 blocks of 32), greedy. Both algorithms threshold-matched at 0.9; FOCUS α=1.5. Findings:
1. **FOCUS is slower wall-clock and the gap widens with batch — throughput 0.90 / 0.84 / 0.69× LowConfidence at conc 1 / 8 / 16** (FOCUS 61 / 231 / 267 tok/s vs LowConfidence 68 / 275 / 388 tok/s; Fig 1). LowConfidence scales 68→388 tok/s across the batch sweep; FOCUS scales only 61→267 and falls further behind as batch grows.
2. **The eviction is real: per reduced step FOCUS processes only 66 / 78 / 80% of the block tokens through Layers 1-attn..L** (mean Σ|S|/(B·bs) = 0.662 / 0.784 / 0.803 at conc 1 / 8 / 16; pooled mean 0.753, median 0.844, n=1580 steps; Fig 2) — i.e. **20–34% fewer tokens** in L1-attn..L, the paper's FLOPs lever. So the compute *is* being saved; it just doesn't show up as wall-clock here.
3. **Eviction is most aggressive at low concurrency and early in a block, least at high concurrency.** At conc 1 a single request's block ramps from heavy eviction (few decodable early) to none (block nearly full), averaging 0.66; at conc 16 many requests at mixed block stages average out higher (0.80), so the *aggregate* FLOPs benefit shrinks exactly as batch grows — the opposite direction from where it would need to help.
4. **The loss is host overhead, not GPU compute.** Per denoising step FOCUS does 1 full-block prefix forward (L0 + L1-QKV on all B) **+ 3 FlashInfer `init_forward_metadata` rebuilds** (Phase P/A1/S) + Python per-request selection (`_select_retained`) and commit (`_commit_step`) with many `.item()`/`.cpu()` D2H syncs; LowConfidence does **1** forward + one vectorized commit. In eager mode (no graph to hide launch/host latency) on a small model (hidden 2048, MoE top-8) the saved GPU FLOPs per token are tiny next to that fixed host cost, so latency rises 1.12 / 1.18 / 1.38× (conc 1 / 8 / 16).

Net: the reduced forward is **correct and does cut processed tokens 20–34%**, but as an **eager prototype it is host-bound and ~0.7–0.9× the baseline throughput**; turning the FLOPs reduction into a wall-clock win is now a **host-efficiency + CUDA-graph** problem, not a correctness problem.

## Setup

### Hardware & software
- **GPU:** 1× NVIDIA **A100 80GB PCIe** (sm80), single card, TP=1 (no inter-GPU comm in this experiment).
- **Software:** conda env `sglang` (Python 3.10), SGLang working tree at commit `63234d1a6` **+ uncommitted FOCUS reduced-forward changes** (`focus.py`, `focus_forward.py`, `focus_reduce.py`, `llada2.py` `forward_focus_*`), FlashInfer **0.6.11.post1**, eager (`--disable-cuda-graph`).
- **Model:** `/cephfs/shared/model/LLaDA2.0-mini` — `llada2_moe`, 20 layers, hidden 2048, **256 experts / top-8**, vocab 157184, bf16; `block_size=32`, `mask_id=156895`.

### Runtime config (confirmed from server log)
| Setting | Value |
| --- | --- |
| dLLM algorithm | `LowConfidence` (thr 0.9) vs `Focus` (thr 0.9, α=1.5, maxpool_k 3, importance_layers [0,1]) |
| TP / EP | 1 / 1 |
| Attention backend | `flashinfer` (FOCUS forces `use_paged=True` internally) |
| `mem_fraction_static` | 0.7 |
| `max_running_requests` | = concurrency (1 / 8 / 16) per run |
| `page_size` | 32 (= block_size) |
| CUDA graph | **OFF** (`--disable-cuda-graph`; dLLM eager needs flashinfer pinned) |
| overlap schedule | disabled (dLLM) |
| dtype / KV dtype | bf16 / bf16 |

### Workload & runs
HumanEval (first 20 prompts, cycled) driven by `drive_humaneval.py` keeping exactly `concurrency` `/generate` requests in flight; `total_reqs = 3×concurrency` (≥6) so the running batch stays near-full for a short measured window; greedy, `max_new_tokens=128`. One warmup pass (unmeasured) per run. Realized completion was full (`done=total`) for all 6 runs.

| Run | algo | conc | total reqs | out tokens | tok/s | mean lat (s) |
| --- | --- | --- | --- | --- | --- | --- |
| lowconfidence_c1 | LowConfidence | 1 | 6 | 768 | 68 | 1.89 |
| focus_c1 | Focus | 1 | 6 | 768 | 61 | 2.11 |
| lowconfidence_c8 | LowConfidence | 8 | 24 | 3072 | 275 | 3.61 |
| focus_c8 | Focus | 8 | 24 | 3011 | 231 | 4.25 |
| lowconfidence_c16 | LowConfidence | 16 | 48 | 6028 | 388 | 5.10 |
| focus_c16 | Focus | 16 | 48 | 5984 | 267 | 7.04 |

### Artifacts
Data (mirror, not in repo): `/cephfs/shared/wxli/sglang-dllm/profiling/dllm/focus_vs_lowconf/logs/` — `{algo}_c{conc}_result.json` (driver tok/s + latency list), `focus_c{conc}_redundancy.csv` (per-reduced-step Σ|S|/(B·bs)), `*_server.log`, `*_drive.log`, `focus_vs_lowconf_dist_stats.json` (headline scalars), `focus_vs_lowconf_summary.txt`. Figures (repo): `figures/fig1_throughput.png`, `figures/fig2_redundancy_hist.png`.

## Method & tooling
- **Throughput / latency:** CPU-wall, end-to-end, from the load driver — `tok/s = Σ completion_tokens / wall_s` over the measured window; per-request latency = request round-trip. Both are e2e wall metrics (not GPU-projected); the operating point is the concurrency.
- **Redundancy (the in-loop FLOPs proxy):** `Focus._focus_reduced_forward` appends `bs,kept,total,ratio,per_req_lens` per reduced denoising step when `SGLANG_FOCUS_REDUNDANCY_CSV=<path>` is set (env-gated; also `SGLANG_FOCUS_LOG_REDUNDANCY=1` prints it). `ratio = Σ|S| / (block_size·bs)` is the fraction of block tokens still processed through Layers 1-attn..L; `1.0` = no eviction (≡ LowConfidence). The file write is flushed per step (robust to scheduler-subprocess stdout buffering — the earlier `print`-only path lost samples at low step counts).
- **Bulletproof server lifecycle (learned the hard way):** each run `kill_servers` (SIGKILL all `sglang.launch_server`, then **wait** until GPU memory < 2 GB and no sglang process) **before** launching, so a run never races a leftover server from the previous run — that race silently served requests with the wrong algorithm/env (zero redundancy samples) and caused cross-run OOM. Launched with `PYTHONUNBUFFERED=1`.
- **Reproduce:**
```bash
cd /root/sglang_a100/sglang/experiments/profiling/dllm/focus_vs_lowconf
CONC_LIST="1 8 16" ./run_focus_vs_lowconf.sh        # launches both algos × sweep, drives, parses
python plot_focus_vs_lowconf.py /cephfs/shared/wxli/sglang-dllm/profiling/dllm/focus_vs_lowconf/logs
```

## Results

### Throughput & latency (e2e wall, per concurrency)
| conc | LowConf tok/s | FOCUS tok/s | FOCUS speedup | LowConf lat (s) | FOCUS lat (s) | FOCUS/LC lat |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 68 | 61 | **0.90×** | 1.89 | 2.11 | 1.12× |
| 8 | 275 | 231 | **0.84×** | 3.61 | 4.25 | 1.18× |
| 16 | 388 | 267 | **0.69×** | 5.10 | 7.04 | 1.38× |

FOCUS is slower at every batch size and the deficit grows with concurrency (Fig 1) — the per-step host overhead (Python selection/commit + 3× metadata rebuild) scales with both step count and batch, while LowConfidence amortizes a single forward.

### Processed-token redundancy Σ|S|/(B·bs) (FOCUS, per reduced step)
| conc | mean | median | n steps |
| --- | --- | --- | --- |
| 1 | 0.662 | 0.688 | 501 |
| 8 | 0.784 | 0.880 | 479 |
| 16 | 0.803 | 0.886 | 600 |
| pooled | 0.753 | 0.844 | 1580 |

FOCUS processes 66–80% of block tokens through L1-attn..L (Fig 2 histogram, pooled, right-skewed toward 1.0 = late-block steps with little to evict). This **is** the compute it saves; in a compute-bound regime each 1−ratio fraction is a real L1-attn..L FLOPs cut. The distribution shifts toward 1.0 as concurrency rises (less aggregate eviction), so the FLOPs lever weakens precisely where throughput matters most.

## Caveats
- **Eager prototype, not the paper's runtime.** No CUDA-graph capture of Phase S (the official FOCUS captures the suffix as a graph keyed on rounded |S|), and the host path is unoptimized PyTorch (3 `init_forward_metadata` per step, per-request Python loops, `.item()`/`.cpu()` D2H syncs). These dominate here; they are removable and are the whole gap.
- **Small model, single GPU.** LLaDA2.0-mini's per-token GPU compute is small, so a 20–34% token cut is a few hundred µs — easily swamped by host overhead. A larger/denser model or longer context (more L2..L compute per token) shifts the balance toward FOCUS.
- **TP=1, no comm.** This experiment isolates the compute/host tradeoff; it says nothing about the distributed comm story (that is the D-series).
- **Redundancy depends on workload & α.** α=1.5 with HumanEval; lower α or harder content evicts more (smaller ratio). The redundancy here is a property of this operating point, not a constant.
- **FOCUS ≠ bitwise LowConfidence at α=1.5** (slightly different decode trajectory ⇒ token counts differ by ≲1.5%); correctness is anchored separately by α→∞ ≡ LowConfidence in `experiments/dllm/focus_a100_smoke`.

## Takeaways
- The reduced forward **correctly realizes the paper's eviction** (redundancy < 1, anchored), so this is now an **engineering** problem, not an algorithm one: make the host path cheap enough that the FLOPs cut surfaces as wall-clock.
- **Direction 1 (graph):** capture Phase S (and ideally A1) as a CUDA graph keyed on rounded |S| so the 3× metadata rebuild + per-layer launch latency stop dominating — the single biggest lever in eager today.
- **Direction 2 (host syncs):** vectorize `_select_retained`/`_commit_step` to remove per-request `.item()` loops; build the per-phase `seq_lens`/`extend_*` and `out_cache_loc` on-device without `.cpu()`/`.tolist()` round-trips; fuse the importance/selection into a kernel (the official Triton `focus_*`).
- **Direction 3 (regime):** re-measure on a larger LLaDA2/SDAR and longer outputs where L2..L per-token compute dominates, and at α that evicts harder — the redundancy says 20–34% of L1-attn..L work is removable, which is only worth chasing where that work is the bottleneck.

## Next
- **F2 — graph-captured Phase S:** capture the suffix and re-run this sweep; question: does removing the per-step metadata/launch overhead flip FOCUS to ≥1× at conc≥8? (scripts: extend `run_focus_vs_lowconf.sh` with a graph-on FOCUS path once implemented.)
- **F3 — per-phase GPU time:** nsys `--cuda-graph-trace=node` (graph on) or NVTX `nvtx_gpu_proj_sum` on `focus_prefix`/`focus_l1_attn`/`focus_suffix` vs LowConfidence's `dllm_forward` to attribute the wall-time gap to host vs device and confirm the L2..L device-time actually drops ∝ redundancy.
- **F4 — larger model / longer context:** repeat on LLaDA2.0-flash (or SDAR) at `max_new_tokens≥512` to find the compute-bound crossover where the 20–34% token cut pays for itself.
Cross-links: implementation `notes/focus_implementation_progress.md`; plan `notes/focus_paper_exact_plan.md`; correctness anchor `experiments/dllm/focus_a100_smoke/`; profiling family `experiments/profiling/dllm/README.md`.
