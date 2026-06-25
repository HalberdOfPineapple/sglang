# dLLM Baseline Distributed-Efficiency Profiling Plan (this repo, stock LowConfidence)

## 0. Scope

Profile the **distributed performance/efficiency** of the *unmodified* dLLM path in this checkout — `LowConfidence` over LLaDA2.0-mini under TP/EP/DP — with **no** streaming / draft / selective-recompute (those live in the other repo). The goal is to characterize how the stock denoising loop behaves as a *distributed* workload, so we know which serving amplifier dominates before any project change lands. Companion: `notes/dllm_distributed_optimization_directions.md` (directions I1–I5), `notes/llada2_workflow_and_parallelism.md` (TP/EP/PP/DP call chain).

**Three dLLM-specific distributed facts drive every experiment** (`llada2_workflow_and_parallelism.md` §dLLM Notes, `server_args.py:4401`):

1. **Collectives fire once per denoising step, not per output token.** A block emitting `B=32` tokens over `S_k` steps pays `S_k` rounds of TP all-reduce + EP all-to-all (+ DP gather/scatter). Total comm ≈ `S_k × comm_per_step` — an `S_k`× amplification vs the AR intuition.
2. **Overlap scheduling is force-disabled for dLLM**, so *all* that comm is **exposed** (no compute to hide it behind). This is the opposite of AR decode, where overlap hides most comm.
3. **The full block is forwarded every step** (no eviction in baseline), so the EP all-to-all moves `B·batch` tokens every step — the all-to-all *volume* is `S_k`× larger than AR for the same output.

Net: distributed cost is dominated by `S_k × (exposed collective time)`, and `S_k` is data-dependent (1–32). That product, decomposed by collective type and by parallelism shape, is what this plan measures. Everything is runnable on the stock repo; `S_k` is the only quantity needing a light counter (§3).

## 1. Setup and the parallelism matrix

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1
export DATA_ROOT=/cephfs/shared/wxli/sglang-dllm   # data tree mirrors repo experiments/
export SGLANG_TORCH_PROFILER_DIR=$DATA_ROOT/profiling/dllm/profiles
```

Base server (from `[[llada2-launch-config-a100]]`, needs `[[a100-sm80-flashinfer-topk-fallback]]`):

```bash
HF_HUB_DISABLE_XET=1 python -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini --dllm-algorithm LowConfidence \
  --host 0.0.0.0 --port 30000 --trust-remote-code \
  --tp-size 4 --ep-size 4 --mem-fraction-static 0.7 --max-running-requests 4
```

**Shapes to sweep** (one server each; record GPUs used):

| Tag                                                          | Flags                                                      | Isolates                                 |
| ------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| TP2                                                          | `--tp-size 2 --ep-size 2`                                  | scaling baseline                         |
| TP4                                                          | `--tp-size 4 --ep-size 4`                                  | reference                                |
| TP8                                                          | `--tp-size 8 --ep-size 8` (if 8 GPU)                       | strong scaling, comm growth              |
| TP4-noEP                                                     | `--tp-size 4` (no `--ep-size`, MoE local/TP)               | EP all-to-all vs TP all-reduce isolation |
| DPATT                                                        | `--tp-size 8 --dp-size 2 --enable-dp-attention` (if 8 GPU) | DP-attention gather/scatter cost         |
| Note: under TP/EP all ranks run the *same* denoising schedule on the same batch; the only place ranks genuinely diverge is **MoE expert routing under EP** (D4). PP is forced to 1 for dLLM, so it is out of scope. |                                                            |                                          |

**Profiling tools:** `nsys profile -t cuda,nvtx,nccl` wraps the launcher and captures all TP/EP child processes; `/start_profile` (`http_server.py:947`) for the torch profiler; `nvidia-smi dmon`; load generators `bench_serving.py` / `bench_one_batch_server.py`.

## 2. Core distributed experiments

### D1 — Per-step comm decomposition & exposed fraction ★ (headline)

Trace the steady state and split each denoising step into compute vs exposed comm, and comm into TP-`AllReduce` / EP-`AllToAll`(DeepEP dispatch+combine) / DP gather-scatter.

```bash
nsys profile -t cuda,nvtx,nccl -o outputs/d1_tp4 --force-overwrite true \
  --capture-range=cudaProfilerApi \
  python -m sglang.launch_server ...TP4...   # then drive steady load, /start_profile to bound
```

From the report: per step, `comm_exposed_ms / step_ms`, and the TP/EP/DP split. **Insight:** the single most important distributed number — what fraction of wall-clock is exposed collectives, and which collective dominates. Gates I3 (state-dependent overlap) and parallelism-shape choice. Because overlap is disabled (fact 2), expect exposed fraction to be high; quantify it.

### D2 — `S_k`× amplification: comm per output token vs AR ★

Combine D1's `comm_per_step` with **`S_k` (`number of diffusion steps spent on block `k) and output length**. Compute `comm_per_output_token = (S_k × comm_per_step) / tokens_committed`. 

**Insight:** sizes the *structural* distributed penalty of diffusion decoding — how many extra collective rounds we pay per delivered token vs an AR model of the same size. This is the headline motivation number for any "fewer steps" (Design 1) or overlap (I3) work, expressed in comm terms rather than FLOPs.



### D3 — Strong scaling & parallelism-shape sweep ★

Run identical load (`bench_serving --dataset-name random --random-input-len 256 --random-output-len 256 --num-prompts 64 --max-concurrency 4`) across TP2/TP4/TP8, log latency, tokens/s, tokens/s-per-GPU, and D1's exposed-comm fraction per shape.

```bash
for tag in tp2 tp4 tp8; do  # relaunch per shape
  python -m sglang.bench_serving --backend sglang --port 30000 \
    --dataset-name random --random-input-len 256 --random-output-len 256 \
    --num-prompts 64 --max-concurrency 4 --output-file outputs/d3_$tag.jsonl
done
```

**Insight:** does adding GPUs actually cut latency, or does the `S_k`×-amplified exposed comm eat the gain? If tokens/s-per-GPU **falls** faster than for AR as TP grows, diffusion is comm-scaling-limited → I3/Design-1 are the levers. The TP4-noEP point isolates how much of the loss is EP all-to-all.

### D4 — EP all-to-all: volume, exposed time, expert-load imbalance ★

From D1's trace under TP4 (EP) vs TP4-noEP: isolate DeepEP dispatch/combine time and bytes per step. Additionally, log per-step **expert token counts** to see if routing imbalance changes as masked positions resolve across the `S_k` steps (the block content changes step-to-step → routing drifts even though token *count* is constant). Lightweight hook in the MoE gate (`models/llada2.py:154`) or read DeepEP dispatch counts. **Insight:** EP all-to-all is the prime candidate for the dominant collective (fact 3: full-block volume every step). Imbalance across steps → a load-balancing opportunity (part of I3); stable balance → focus on raw all-to-all volume reduction. The one genuinely rank-divergent behavior in the stock path.

### D5 — DP-attention gather/scatter cost in the loop

Only if 8 GPU. Compare DPATT vs TP8 at equal GPUs: D1 exposed-comm fraction and latency. The DP-attention path adds per-layer gather/scatter/reduce-scatter (`layers/dp_attention.py:534+`) that is paid **every denoising step**. **Insight:** whether DP-attention helps or hurts dLLM specifically (it may be net-negative here because its layout comm is `S_k`-amplified, unlike AR). Decides whether DP-attention belongs in the dLLM serving recipe.

### D6 — Exposed-comm / overlap headroom

From D1, compute `overlap_headroom = exposed_comm_ms` (since overlap is off, all of it is potentially hideable). Optionally A/B against AR-style overlap by checking whether re-enabling overlap is even legal here (`server_args.py:4401` forces it off — confirm it cannot, and note *why*: the loop mutates `input_ids` mid-step). **Insight:** the upper bound on what I3 (state-dependent overlap) can recover — and the evidence that plain overlap is blocked by the dLLM loop structure (making the workaround non-trivial, per the engineering-vs-innovation test).

### D7 — Distributed straggler / batch heterogeneity

With `--max-running-requests 4` and a **mixed** easy/hard prompt set, the loop runs until *all* blocks in the batch are mask-free (`low_confidence.py:52`), so every rank keeps doing full-batch forwards + collectives for already-finished rows. Log (via §3 counter) per request the step it finished vs the batch-exit step; multiply wasted steps by D1's `comm_per_step`. **Insight:** the **wasted collective rounds** caused by batch heterogeneity — a distributed-cost framing of the straggler problem that motivates I1/I5 (ragged batching / throughput packing). Larger batch ⇒ worse, so also sweep concurrency 2/4/8.

### D8 — Per-rank memory & KV footprint

Per shape, log per-GPU memory (`nvidia-smi`), the KV pool size SGLang reports at startup, and the effect of `page_size==block_size` (forced for dLLM). **Insight:** how TP/EP split the 256-expert MoE residency and KV across ranks, and whether the 32-token page granularity wastes KV under distributed batching. Feeds memory/cache directions and the `mem-fraction-static` headroom story from `[[llada2-launch-config-a100]]`.

### D9 — (if multi-node) inter-node collective cost

If the allocation spans nodes, repeat D1 with `--nnodes 2`. Inter-node NCCL is far slower, and the `S_k`× amplification makes it brutal. **Insight:** whether dLLM serving is viable multi-node at all without overlap/step-reduction, or whether it must stay single-node — a hard constraint for the project's distributed framing.

## 3. Minimal supporting instrumentation (only to interpret the above)

The distributed experiments need `S_k` (steps per block) and finish-step-per-request, which are not in `meta_info`. Add a **single counter** behind `SGLANG_DLLM_PROFILE=1`, confined to `low_confidence.py` — far lighter than the algorithm-characterization instrumentation in earlier drafts:

- Count loop iterations `S_k` until break (`low_confidence.py:51–53`).
- Per request, record the iteration at which its block became mask-free (`:66` block test) and the batch-exit iteration (`:53`).
- Append one CSV row per block (`block_id, S_k, per_req_finish_steps, n_committed`) under `$DATA_ROOT/profiling/dllm/`.
  This is profiling-only and reversible; the baseline path is bit-identical when the flag is unset (CLAUDE.md isolation). `comm_per_step` and the TP/EP/DP split come entirely from nsys (no source edit).

## 4. What each experiment decides

| Exp  | Tool             | Distributed question                           | Direction                               |
| ---- | ---------------- | ---------------------------------------------- | --------------------------------------- |
| D1   | nsys NCCL        | exposed-comm fraction & TP/EP/DP split         | headline; I3                            |
| D2   | nsys + `S_k`     | comm-per-output-token vs AR (`S_k`× penalty)   | Design 1, I3 motivation                 |
| D3   | bench sweep      | strong scaling; does GPU count help?           | parallelism shape, I3                   |
| D4   | nsys + gate hook | EP all-to-all volume & load drift across steps | I3 (load balance), all-to-all reduction |
| D5   | bench + nsys     | DP-attention net effect in the loop            | serving recipe                          |
| D6   | nsys             | overlap headroom; why plain overlap is blocked | I3 (non-trivial workaround)             |
| D7   | `S_k` counter    | wasted collective rounds from stragglers       | I1, I5                                  |
| D8   | nvidia-smi       | per-rank KV/MoE residency, page granularity    | memory/cache                            |
| D9   | nsys multi-node  | inter-node viability                           | distributed constraint                  |

## 5. Suggested first pass

1. Launch **TP4**; smoke test; turn on the `S_k` counter (§3).
2. **D1** → exposed-comm fraction and TP-vs-EP split (the headline).
3. **D2** → `S_k`× comm-per-token penalty (the motivation number).
4. **D3** (TP2/TP4[/TP8]) → strong scaling; is diffusion comm-scaling-limited?
5. **D4** → confirm whether EP all-to-all is the dominant collective.
   Defer D5/D7/D8/D9 until the headline says where the distributed cost concentrates.

## 6. Hygiene

- Scripts in-repo under `experiments/profiling/dllm/`; all outputs/nsys reports under the mirrored data tree `$DATA_ROOT/profiling/dllm/{profiles,logs}/` (`DATA_ROOT` default `/cephfs/shared/wxli/sglang-dllm`); never `/tmp` or `/root` (AGENTS.md) — nsys reports are large, keep on CephFS.
- Record per run: git commit, model, GPU count/type, TP/EP/DP shape, `mem-fraction-static`, concurrency, prompt file. Summarize as `notes/experiment_YYYYMMDD_baseline_dllm_distributed.md`.
- nsys-profile the launcher so all TP/EP ranks are captured; bound the capture with `/start_profile`+`/stop_profile` to keep traces small.
- Keep the §3 counter behind `SGLANG_DLLM_PROFILE`; baseline path must be bit-identical when unset.