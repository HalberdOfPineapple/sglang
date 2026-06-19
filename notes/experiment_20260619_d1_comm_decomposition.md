# Experiment D1 — Per-step Comm Decomposition (baseline LowConfidence, 4×A100)

## Summary

First distributed profiling pass on the **stock** dLLM path (no project changes), bs=1. All per-phase numbers use **GPU time projected onto NVTX ranges** (`nvtx_gpu_proj_sum`), not CPU push/pop wall-time — see the *Metric* note below, this distinction changed the conclusions. Three findings:

1. **At bs=1 the production (CUDA-graph) path is host-bound, not comm- or compute-bound.** Per denoising step the GPU is busy only **~1.41 ms** (forward incl. all collectives 0.85 ms + selection 0.56 ms), yet the step takes **~17 ms wall-clock** — the `dllm_select` phase is **15.5 ms of CPU** vs 0.56 ms of GPU. The host-side Python selection (repeated `.item()` syncs + full-vocab softmax/argmax/topk launches) dominates wall-clock; **GPU idle ~92% of the step**.
2. **Communication is cheap in the production path but explodes ~64× when the CUDA graph is disabled** — and that explosion is **NCCL spin-wait, not transfer.** Graph forward = 0.85 ms GPU; eager forward = **54.7 ms** GPU, ~62% of it RING_LL `AllReduce` (~40 per forward, ~0.7 ms each) + a 5.6 ms full-vocab `AllGather`. Eager launches collectives one-at-a-time so CPU-launch jitter desynchronizes the ranks and each all-reduce *spins* waiting for peers; the graph replays the forward in lockstep across all 4 ranks, so peers arrive together and spin ≈ 0. **⚠️ The earlier "87% comm" headline was an eager-mode artifact — it overstates production comm cost.**
3. **The TP=4 ring straddles two NUMA/sockets, so every collective takes the slowest path.** `nvidia-smi topo -m`: GPU0–GPU1 = NODE (NUMA0), GPU2–GPU3 = NODE (NUMA1), but GPU0/1 ↔ GPU2/3 = **SYS** (cross-socket PCIe+UPI, no P2P). The ring all-reduce crosses SYS, which is what makes the eager spin so costly and inflates the AllGather. A **TP=2 confined to one NODE pair** (GPU0+1 or GPU2+3) would avoid the SYS hop → D3 experiment.

Net for the project: at bs=1 the bottleneck is **host-side selection + full-vocab logits**, not the network. The full-vocab logits is doubly implicated — the top eager compute kernel (softmax) *and* the per-forward `AllGather` — so logits sparsification (D2/E-b) helps both. Distributed comm only becomes the headline at larger batch / on the cross-socket topology in eager mode; quantify with D3 (batch sweep + TP-shape vs NUMA) before investing in comm-overlap work.

## Setup

### Hardware & software

- **GPUs:** 4× NVIDIA A100 80GB **PCIe** (no NVLink/NVSwitch). All 4 idle before the run. **Topology (`nvidia-smi topo -m`) matters for comm:** GPU0–GPU1 = `NODE` (same PCIe host bridge, NUMA0, CPU 0-31/64-95); GPU2–GPU3 = `NODE` (NUMA1, CPU 32-63/96-127); **GPU0/1 ↔ GPU2/3 = `SYS`** (cross-socket, traverses PCIe + inter-CPU UPI, no P2P) — the slowest hop. A TP=4 ring spans both sockets and crosses SYS; a TP=2 within one NODE pair would not.
- **Software:** conda env `sglang` (Python 3.10.20), SGLang from this checkout (`/root/sglang_a100/sglang`, git `git rev-parse HEAD` to pin), `nsys` 2026.3.1 at `/usr/local/bin/nsys`.
- **Model:** `inclusionAI/LLaDA2.0-mini` — `model_type=llada2_moe`, 20 layers, hidden 2048, **256 experts / top-8**, vocab **157184**, ~17B total / ~1.4B active, bf16 → ~7.6 GB/GPU at TP4. First launch downloads ~34 GB (`HF_HUB_DISABLE_XET=1` required; see [[hf-download-disable-xet-proxy]]); cached afterward.

### Parallelism / runtime config (as launched)

| Setting | Value | Source |
| --- | --- | --- |
| dLLM algorithm | `LowConfidence`, threshold **0.95** (default, no YAML) | `--dllm-algorithm LowConfidence` |
| TP / EP | **4 / 4** | `--tp-size 4 --ep-size 4` |
| MoE A2A backend | **`none`** (no EP all-to-all; fused-Triton MoE + TP all-reduce) | not passed → default |
| Attention backend | `flashinfer` | dLLM default |
| `mem_fraction_static` | **0.7** | OOM-safe for 256-expert capture, see [[llada2-launch-config-a100]] |
| `max_running_requests` | 4 | `--max-running-requests 4` |
| `page_size` | 32 (= block_size) | dLLM forced |
| overlap schedule | **disabled** | dLLM forced (`disable_overlap_schedule=True`) |
| piecewise CUDA graph | **disabled** | dLLM forced (`disable_piecewise_cuda_graph=True`) |
| full CUDA graph | **ON** in `d1_tp4`, **OFF** in `d1_tp4_eager` | default vs `--disable-cuda-graph` |

Confirmed from the server log: `Capture cuda graph bs [1, 2, 4]`, `tp_size=4`, `ep_size=4`, `moe_a2a_backend='none'`, `disable_overlap_schedule=True`. Requires the [[a100-sm80-flashinfer-topk-fallback]] patch (already in this tree).

### Workload

- 1 warmup `/generate` (**outside** the capture window) to finish weight load + CUDA-graph capture + first-touch allocations.
- 2 profiled `/generate` requests (a math-reasoning and a code prompt), `max_new_tokens=128`, `temperature=0`, sent **sequentially** via blocking `curl` → effective **batch = 1** per forward → a **latency-bound** regime. 128 tokens ≈ 4 blocks × 32, and each block runs `S_k` denoising steps.

### Two runs (only knob changed = CUDA graph)

| Tag | CUDA graph | Purpose |
| --- | --- | --- |
| `d1_tp4` | ON (production) | what the real serving path looks like; comm is *inside* the graph |
| `d1_tp4_eager` | OFF (`--disable-cuda-graph`) | expose graph-internal NCCL kernels so comm is attributable |

Artifacts live in a data tree that **mirrors the repo `experiments/` hierarchy**: `DATA_ROOT/profiling/dllm/` with `DATA_ROOT` default `/cephfs/shared/wxli/sglang-dllm`. So profiles are in `…/profiling/dllm/profiles/`: `<tag>.nsys-rep`, `<tag>.sqlite`, `<tag>_cuda_gpu_kern_sum.csv`, `<tag>_nvtx_pushpop_sum.csv`, `<tag>_cuda_gpu_trace.csv`; server/stats logs in `…/profiling/dllm/logs/<tag>_server.log`.

## Method & tooling (what was reused vs added)

### Instrumentation added (minimal, dLLM **algorithm** module only — the model module `llada2.py` is untouched)

1. **`python/sglang/srt/dllm/profiling.py`** (new) — env-gated NVTX helpers, **no-op unless `SGLANG_DLLM_NVTX=1`** so the baseline path is bit-identical when off:

   ```python
   DLLM_NVTX_ENABLED = os.environ.get("SGLANG_DLLM_NVTX", "0") == "1"
   @contextmanager
   def dllm_nvtx_range(name):           # nvtx.range_push/pop when enabled, else pass
       ...
   def dllm_nvtx_push(name): ...        # for ranges that span a loop body
   def dllm_nvtx_pop(): ...
   ```

   Chosen over `--enable-layerwise-nvtx-marker` (which wraps *every* `nn.Module` via `utils/nvtx_pytorch_hooks.PytHooks`) because that is far too fine-grained and hides the denoising-loop structure we want.
2. **`python/sglang/srt/dllm/algorithm/low_confidence.py`** — wrapped the loop phases (`run()`):
   - fast path → `dllm_prefill_forward`
   - per step `N`: `with dllm_nvtx_range(f"dllm_forward.step{N}")` around `model_runner.forward(...)`; `dllm_nvtx_push(f"dllm_select.step{N}")` … `dllm_nvtx_pop()` around the host argmax/softmax/threshold/commit block
   - final forward → `dllm_final_forward`

   So NCCL kernels emitted during the model forward land inside `dllm_forward.step{N}`, and the host token-selection cost lands inside `dllm_select.step{N}`.

### Capture bounding reused from SGLang (no new infra)

SGLang's profiler accepts a **`CUDA_PROFILER`** activity that calls `torch.cuda.cudart().cudaProfilerStart()/Stop()` on the base rank (`managers/scheduler_profiler_mixin.py:213` / `:323`). nsys `--capture-range=cudaProfilerApi` keys off exactly those calls, so the capture window is driven over HTTP — no extra hooks, no torch profiler. nsys auto-tags NCCL kernels (`ncclDevKernel_AllReduce_*`, `_AllGather_*`), so collective-type decomposition is free.

### Exact reproduction

Driver: `experiments/profiling/dllm/run_d1.sh` (in-repo; outputs go to `/cephfs` via `OUT`). Environment + the two runs:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sglang
export HF_HUB_DISABLE_XET=1 NO_PROXY=localhost,127.0.0.1
# Run A — production (CUDA graph ON)
TAG=d1_tp4        bash experiments/profiling/dllm/run_d1.sh
# Run B — eager (exposes graph-internal NCCL)
TAG=d1_tp4_eager EXTRA_ARGS="--disable-cuda-graph" READY_TIMEOUT=600 \
                  bash experiments/profiling/dllm/run_d1.sh
```

What `run_d1.sh` does, step by step (key commands inlined):

```bash
export SGLANG_DLLM_NVTX=1                         # enable the NVTX ranges

# 1) launch the server *under* nsys (background). capture-range=cudaProfilerApi
#    => nsys records only between cudaProfilerStart and cudaProfilerStop.
#    stop-shutdown => end collection AND kill the server after /stop_profile.
nsys profile -t cuda,nvtx,nccl \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --force-overwrite true -o "$REP" \
  python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini --dllm-algorithm LowConfidence \
    --host 0.0.0.0 --port 30000 --trust-remote-code \
    --tp-size 4 --ep-size 4 --mem-fraction-static 0.7 --max-running-requests 4 \
    $EXTRA_ARGS &

# 2) wait until ready (polls /get_model_info; first run also downloads weights)
until curl -s --noproxy '*' http://127.0.0.1:30000/get_model_info >/dev/null; do sleep 5; done

# 3) warmup (NOT captured — capture only starts at cudaProfilerStart)
curl -s --noproxy '*' http://127.0.0.1:30000/generate -d '{"text":"...","sampling_params":{"max_new_tokens":128,"temperature":0}}'

# 4) START capture -> run profiled workload -> STOP capture
curl -s --noproxy '*' -X POST http://127.0.0.1:30000/start_profile \
     -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}'
curl ... /generate  (math prompt) ; curl ... /generate  (code prompt)
curl -s --noproxy '*' -X POST http://127.0.0.1:30000/stop_profile   # -> nsys finalizes + shuts down

# 5) post-process to CSV (then parse_d1.py summarizes)
nsys stats --force-export=true \
  --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,nvtx_pushpop_sum,cuda_gpu_trace \
  --format csv --output "$REP" "${REP}.nsys-rep"
```

For comm's share of E2E with the graph **on**, add `--cuda-graph-trace=node` to the `nsys profile` line so in-graph NCCL kernels are recorded (see methodology §5).

### nsys statistics & post-processing methodology

This is the authoritative description of how raw nsys output is turned into the numbers in *Results*. All of it is implemented in `experiments/profiling/dllm/parse_d1.py` (run automatically at the end of `run_d1.sh`).

**1. Reports generated.** `nsys stats --force-export=true --format csv` produces one CSV per report from `<REP>.nsys-rep` (which is also exported to `<REP>.sqlite`):

| Report | CSV | What it gives | Used for |
| --- | --- | --- | --- |
| `cuda_gpu_kern_sum` | `<REP>_cuda_gpu_kern_sum.csv` | per-kernel total GPU time, instances, name | comm-vs-compute split, op mix, top kernels |
| `nvtx_gpu_proj_sum` | `<REP>_nvtx_gpu_proj_sum.csv` | **GPU time projected onto each NVTX range** (`Total Proj Time`) + CPU range time (`Total Range Time`) | true per-phase *device* cost (forward vs select) |
| `nvtx_pushpop_sum` | `<REP>_nvtx_pushpop_sum.csv` | CPU push/pop wall-time per range | per-phase *host/wall* cost only |
| `cuda_gpu_trace` | `<REP>_cuda_gpu_trace.csv` | per-launch GPU trace (start/dur/stream) | timeline gap analysis (true *exposed* comm) |

**2. CSV parsing.** Kernel names contain commas (C++ templates), so CSVs **must** be read with `csv.DictReader` — naive `split(',')` corrupts columns. The canonical kernel-name list also lives in `<REP>.sqlite` (`SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id`).

**3. Comm vs compute (`cuda_gpu_kern_sum`).** Total GPU time `T = Σ Total Time (ns)` over all kernels. A kernel is **COMM** iff its `Name` matches the regex `nccl|all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|cross_device|nvshmem|deep.?ep|dispatch|combine|sendrecv|broadcast` (case-insensitive), else **COMPUTE**. Comm grouped by op: `…AllReduce…`→TP, `…AllToAll…`→EP, `…AllGather…`→vocab/LM-head, etc. Per-op % = op time / `T`.

**4. Per-phase device cost (`nvtx_gpu_proj_sum`) — the metric used in Results.** For each phase prefix (`dllm_forward.step`, `dllm_select.step`, `dllm_final_forward`, `dllm_prefill_forward`): `GPU = Σ Total Proj Time (ns)`, `CPU = Σ Total Range Time (ns)`, `inst = Σ Range Instances`; report `GPU/inst` and `CPU/inst`. **`nvtx_gpu_proj_sum` is preferred over `nvtx_pushpop_sum`** because push/pop is CPU wall-time and misattributes async/`.item()` stalls across phases (see *Metric* note); push/pop is reported only as host/wall cost.

**5. Communication fraction of E2E (the headline metric to use going forward).** Exposed-comm fraction `= (Σ NCCL kernel GPU time) / (E2E inference time)`. Two requirements for it to be meaningful:
- **CUDA graph ON** (production). By default nsys traces a CUDA graph as one opaque node, so in-graph NCCL is invisible in `cuda_gpu_kern_sum`. Add **`nsys profile --cuda-graph-trace=node`** so each kernel *inside* the graph (incl. NCCL) is recorded individually — this exposes collectives **without** disabling the graph (avoids the eager spin-wait distortion entirely).
- **GPU-bound operating point** (batched concurrency), not bs=1 — at bs=1 the loop is host-bound, so comm/E2E is near-zero and unrepresentative.
- Because dLLM disables overlap scheduling, NCCL kernel time is genuinely *exposed* (serialized with compute), so the kernel-time ratio is a fair exposed-comm fraction. For the denominator, use the captured window's wall-clock (from `/start_profile`→`/stop_profile`) or the per-request E2E latency from `bench_serving`.

**6. Derived sanity checks.** Forwards ≈ AllReduce-instances ÷ (num_layers × 2 collectives/layer); per-step GPU total = (forward GPU + select GPU)/steps. Cross-check phase sums against `cuda_gpu_kern_sum` total.

## Results

### Metric: GPU-projected vs CPU-range (read this first)

Per phase, two numbers from `nsys stats`: **`nvtx_gpu_proj_sum` → GPU time projected onto the range** (true device work), and **`nvtx_pushpop_sum` → CPU push/pop wall-time**. They diverge sharply and the CPU number alone is *misleading*: the forward launches a CUDA-graph replay asynchronously (CPU returns in ~1.4 ms while GPU work is still queued), and the first `.item()` in `dllm_select` then blocks and *absorbs* that queued GPU time — so CPU-range makes select look 10× the forward when the actual GPU work is the opposite. **All conclusions below use GPU-projected time; CPU-range is reported only as wall-clock/host cost.**

### Run d1_tp4 (CUDA graph ON — production path)

Per-step, per call (`nvtx_gpu_proj_sum`):

| Phase | **GPU/step** | CPU-range/step | note |
| --- | --- | --- | --- |
| `dllm_forward` (graph replay, incl. all collectives) | **0.85 ms** | 1.46 ms | true device work small |
| `dllm_select` (host) | **0.56 ms** | **15.47 ms** | GPU tiny; CPU is `.item()`+Python |
| **per-step total** | **1.41 ms GPU** | **~17 ms wall** | **GPU idle ~92% of step** |

- **Host-bound at bs=1:** the step is ~17 ms wall for ~1.41 ms of GPU. The `dllm_select` CPU cost (15.5 ms) is the **Python selection loop + repeated `.item()` syncs** (lines 62/77/97) + many small kernel launches; the *GPU* part of selection is only 0.56 ms.
- The `cuda_gpu_kern_sum` for this run lists **only the eager selection kernels** (forward kernels are inside the graph): largest is `cunn_SoftMaxForward` = 39.4 ms total / 488 = 80 µs each = `F.softmax(curr_logits)` over the **157k vocab** per block-position per step, then `reduce`/`argmax`/cub-scan/select (the `topk`/`.item()` machinery). This is the full-vocab logits cost on the GPU side.
- Step-instance histogram decays 36 (steps 0–7) → 4 (step 24): **front-loaded decode yield** (deck §3.1) confirmed — many blocks need 8–25 denoising steps.

### Run d1_tp4_eager (CUDA graph OFF — exposes collectives, but changes their cost)

Per-step GPU-projected: `dllm_forward` = **54.7 ms**, `dllm_select` = 0.89 ms (per-step total 55.6 ms GPU, forward 98%). Whole-run `cuda_gpu_kern_sum` = 20,795 ms:

| Class | Time | % of GPU |
| --- | --- | --- |
| **COMM** | 18,150 ms | **87.3%** |
| ↳ `AllReduce_Sum_bf16_RING_LL` | 15,177 ms / 21,648 inst (~0.70 ms each) | 73.0% |
| ↳ `AllGather_RING_LL` | 2,973 ms / 528 inst (~5.6 ms each) | 14.3% |
| **COMPUTE** | 2,645 ms | 12.7% |
| ↳ `fused_moe_kernel` | 853 ms | 4.1% |

Interpretation — **eager forward is 64× the graph forward (0.85 → 54.7 ms), and that gap is collective spin-wait, not compute or transfer:**

- **~21,648 all-reduces** ≈ 40 per forward (20 layers × {attn-out, MoE-out}) × ~541 forwards → the per-step collective structure is real (`S_k`× amplification).
- Eager launches each collective separately, so CPU-launch jitter **desynchronizes the 4 ranks** and each `RING_LL` all-reduce *spins* on the GPU waiting for peers (~0.70 ms each). The CUDA graph replays the whole forward in **lockstep across ranks** → peers arrive together → spin ≈ 0, which is why the graph forward is only 0.85 ms. **So eager is not a faithful proxy for production comm cost; "87% comm" is the cost of *removing the graph*, not the cost paid in serving.**
- The spin is amplified by **topology**: the TP=4 ring crosses the cross-socket `SYS` link (see Setup), the worst hop. The `AllGather` at 5.6 ms/call is the **vocab-parallel LM-head full-logits gather** (157k vocab across the SYS boundary) — same full-vocab root cause as the softmax kernel.

## Caveats (read before generalizing)

- **Use GPU-projected (`nvtx_gpu_proj_sum`), not CPU push/pop, for per-phase cost.** CPU-range attributes a `.item()` stall to whichever phase issues it (graph-ON inflates `select`, graph-OFF inflates `forward`); it's only valid as wall-clock/host cost, not isolated device work. (See *Metric* note.)
- **Eager (`--disable-cuda-graph`) inflates collective cost via spin-wait** and is therefore *not* a faithful measure of production comm. It is useful only to (a) confirm collectives exist and their op-mix, (b) bound the worst case. Production comm at bs=1 is ~0.85 ms/forward (graphed), small.
- **`RING_LL` duration includes spin-wait** → comm % in the eager run is GPU-occupancy, dominated by sync/latency, not transfer bandwidth.
- **Topology-bound:** TP=4 ring crosses `SYS` (cross-socket). Numbers would differ on NVLink/NVSwitch or with TP confined to a `NODE` pair. Re-measure on the target interconnect before quoting comm fractions.
- **Single base-rank capture** (`capture-range=cudaProfilerApi`) → one representative TP rank.
- **batch=1** (sequential curl) → host-bound, latency regime. Comm/compute and the host-bound conclusion both shift with batch. (→ D3.)
- **`moe_a2a_backend='none'`** → no EP all-to-all here; EP-dispatch (D4) needs `--moe-a2a-backend deepep` (may not be viable on PCIe/SYS).

## Takeaways for direction priority

- **#1 at bs=1: host-side selection (E-a).** The step is host-bound — ~17 ms wall for 1.41 ms GPU. Removing the `.item()` syncs and the per-block Python loop (fuse argmax/softmax-conf/threshold/commit into one GPU pass) attacks the dominant wall-clock cost directly and is isolated/low-risk.
- **#2: full-vocab logits (D2/E-b).** Doubly implicated — top GPU selection kernel (softmax over 157k vocab) *and* the per-forward `AllGather` (vocab-parallel LM head). Computing logits only for still-masked positions cuts both compute and a collective. Cheap, no model-parallel redesign.
- **Comm is NOT the bs=1 bottleneck; do not over-index on it yet.** In the production graph path the forward incl. all collectives is 0.85 ms. The "87% comm" only appears with the graph disabled (spin-wait) and on the cross-socket SYS topology. I3 (state-dependent overlap) is premature until **D3** shows comm re-emerging at higher batch / different TP-shape.
- The CUDA-graph finding answers an open question in `dllm_distributed_optimization_directions.md` (E-c/D11): **graphing the static-shape forward is already ON** (and is what keeps collectives in lockstep) — that engineering win is spent.

## Next

- **D3 batch sweep** (concurrency 1/2/4/8) graph-ON: does the step stay host-bound, and does GPU/comm time re-emerge as bs grows?
- **D3b TP-shape vs NUMA:** TP=2 on a `NODE` pair (GPU0+1) vs TP=2 across `SYS` (GPU0+2) vs TP=4 — isolate the cross-socket penalty quantified here.
- **D2 probe:** logits/softmax cost (full-vocab) vs masked-position-only logits.
- Optional: timeline gap-analysis (`cuda_gpu_trace`) for true *exposed* comm rather than occupancy sum.
- Scripts: `experiments/profiling/dllm/{run_d1.sh,parse_d1.py}` (in-repo). Data: `d1_tp4*` and `d1_tp4_eager*` under `/cephfs/shared/wxli/sglang-dllm/profiling/dllm/{profiles,logs}/`.