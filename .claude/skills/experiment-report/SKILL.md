---
name: experiment-report
description: Lay out an experiment under experiments/, route scripts (repo) vs data (CephFS mirror) correctly, and write a dense dLLM/SGLang experiment report inside the experiment folder. Use when creating/organizing a profiling or benchmark experiment, storing its scripts/outputs, or writing/updating its report.
---

# Experiment Layout, Storage, and Report Writing

Conventions for this repo's dLLM-on-SGLang experiments. Two inseparable concerns: **where things live** (scripts in repo, data on CephFS, report in the experiment folder) and **how the report reads** (dense, Typora-friendly, metric-disciplined). Canonical examples to copy from: `notes/templates/experiment_report_template.md`; the full reports `experiments/profiling/dllm/d1_comm_decomposition/README.md` and `experiments/profiling/dllm/d2_sk_amplification/README.md`; their scripts `run_d1.sh`/`parse_d1.py` and `run_d2.sh`/`drive_humaneval.py`/`parse_d2.py`/`plot_d2.py`.

## 1. Folder layout — one self-contained folder per experiment

Each experiment is a leaf folder holding **everything for that experiment**: driver script(s), parser(s), and a `README.md` that *is* the report.
```
experiments/<family>/<subject>/<exp>/      e.g. experiments/profiling/dllm/d1_comm_decomposition/
  run_<id>.sh        # single copy-pasteable reproduction entry point
  parse_<id>.py      # turns raw outputs into the reported numbers
  README.md          # the experiment report (see §3)
```
Add a `<family>/<subject>/README.md` index that maps the plan's experiment IDs to folders (see `experiments/profiling/dllm/README.md`), and a top-level `experiments/README.md` stating these conventions. Future experiments get sibling folders — never dump a second experiment's scripts into an existing folder.

## 2. Scripts vs data — repo vs CephFS, mirrored

- **Scripts live in the repo** (the experiment folder). They are small, versioned, reviewable.
- **Data is large and goes to CephFS**, in a tree that **mirrors** the repo `experiments/` hierarchy: a leaf at `experiments/<family>/<subject>/<exp>/` writes to `$DATA_ROOT/<family>/<subject>/<exp>/{profiles,logs}/`. Note the mirror **drops the leading `experiments/`**. Default `DATA_ROOT=/cephfs/shared/$USER/sglang-dllm` (this project uses `/cephfs/shared/wxli/sglang-dllm`).
- **Never** put important outputs (nsys reports, traces, CSVs, logs, checkpoints) in `/tmp`, `/root`, `/home`, `/etc`, `/var`, `/usr` — they are container-temporary (AGENTS.md). nsys reports/traces are huge; keep them on CephFS.
- The run script parameterizes everything via env (`DATA_ROOT`, `OUT`, `PROF`, `LOGS`, `TAG`, `TP`, `EP`, `MEMFRAC`, `CONCURRENCY`, …), `mkdir -p` its output dirs, and records per run: git commit, model, GPU count/type + **interconnect**, TP/EP/DP shape, `mem-fraction-static`, concurrency, prompts.
- **Path gotcha:** when the run script invokes the parser, the repo path is `$REPO/experiments/$EXP_PATH/parse_*.py` while the data path is `$DATA_ROOT/$EXP_PATH/...` — `EXP_PATH` is the mirror suffix (`profiling/dllm/<exp>`), so don't forget the `experiments/` prefix on the repo side.

## 3. The report IS the experiment folder's `README.md`

- Write the **full report inside the experiment folder**. Do **not** split it into a thin `notes/` pointer/stub — that fragments it and is wrong here. (If a dated `notes/` index entry is ever wanted, it is in addition to, never instead of, the folder report; default to folder-only.)
- It must be a **real report**, not a bullet sketch. Follow `notes/templates/experiment_report_template.md` section-for-section and match the depth of the canonical D1 reports:
  - **Title** — `Experiment <ID> — <Title> (<model>, <HW e.g. 4×H100 NVLink>)`.
  - **Summary** — lead with the conclusion and the headline number(s); numbered findings; **flag anything that corrects a prior result**; state the operating point (batch/concurrency) because conclusions depend on it.
  - **Setup** — Hardware & software (topology/NVLink first-class — paste the decisive `nvidia-smi topo -m`/`nvlink` lines; git commit; nsys/lib versions), Model (arch/params), Parallelism/runtime config **table confirmed against the server log**, Workload + the regime it implies, Runs table, Artifacts path.
  - **Method & tooling** — instrumentation (env-gated, isolated so the baseline path is bit-identical when off), capture method, **exact copy-pasteable reproduction**, nsys/post-processing methodology.
  - **Results** — tables first, prose second; **label every metric** (GPU-projected vs CPU-wall, % of GPU vs % of E2E) and state the operating point with every number.
  - **Caveats** — what would change the result (interconnect, batch, graph on/off, single-rank capture, …).
  - **Takeaways** — map findings to the optimization directions; distinguish engineering vs innovation.
  - **Next** — follow-ups with the specific question each answers; point to scripts + data paths.
- **Cross-link** the plan (`notes/dllm_baseline_profiling_plan.md`) and any prior report you are comparing against (e.g. A100 vs H100), with a comparison table when re-running the same experiment on different hardware.

## 4. Formatting — dense, Typora-friendly

The reader uses Typora; optimize the markdown source for it.
- **No mid-paragraph hard wrapping.** Write each bullet and paragraph as a **single flowing line** — do not insert newlines to wrap prose at ~80 cols.
- **Minimal blank lines.** Keep blank lines only where markdown structurally needs them (around headings, and before/after tables and lists). **At most one** consecutive blank line; never double-space. No blank lines between bullets in the same list. (CLAUDE.md: "avoid unnecessary blank lines between main contents.")
- Dense but readable: clear headings, compact bullets, tables for numbers. No filler prose.
- Sanity check the source: `awk 'BEGIN{m=0;c=0}/^$/{c++;if(c>m)m=c;next}{c=0}END{print m}' README.md` should print `1`.

## 5. Metric discipline (carries across all profiling reports)

- **Per-phase device cost: use GPU-projected NVTX time** (`nvtx_gpu_proj_sum` `Total Proj Time`), **not** CPU push/pop (`nvtx_pushpop_sum`) — push/pop is wall-time that misattributes async graph-launch and `.item()` stalls across phases; report it only as host/wall cost.
- **Comm fraction is faithful only with CUDA graph ON + `nsys --cuda-graph-trace=node`, at a GPU-bound operating point.** Without node-trace the graph is one opaque node and in-graph NCCL is invisible. At bs=1 the dLLM loop is host-bound, so comm/E2E is unrepresentative — drive concurrency (≈ `--max-running-requests`) to reach a GPU-bound point.
- **Eager (`--disable-cuda-graph`) inflates collectives via cross-rank spin-wait** — a worst-case bound, **not** a production number; the CUDA graph keeps ranks in lockstep. (dLLM eager also needs `--attention-backend flashinfer` pinned or fa3 crashes — see memory `dllm-eager-needs-flashinfer-attn`.)
- Parse nsys CSVs with `csv.DictReader` — kernel names contain commas; sqlite is the source of truth. Classify comm by op (`AllReduce`→TP, `AllToAll`→EP, `AllGather`→LM-head/vocab). Add derived sanity checks (e.g. forwards ≈ AllReduce_inst / (layers × collectives)).
- **Always state the operating point** (batch/concurrency) next to every number; a metric without it is meaningless.

## 6. Per-token vs per-forward distributions (do not conflate — this is the #1 review trap)

A "per-token" cost is built from **one measurement per forward**: `(comm_time, comp_time, n_tokens_committed)` → per-token sample = `time / n_tokens`. Never approximate it as a block-level step-count × one global average — measure each forward and divide by the tokens *that* forward decoded.
- **Per-forward ≠ per-token as distributions.** One forward decodes many tokens, so in a *per-token* view each forward must be **weighted by its `n` tokens** (a forward decoding 40 tokens represents 40 tokens). Per-forward = each forward one sample. They are different objects; never write "per forward = per token."
- **Ratio of means ≠ mean of ratios.** The token-weighted mean (`Σtime / Σtokens`, the true average cost per delivered token) differs from the unweighted mean of per-forward `time/n` ratios (inflated by 1-token forwards). State which you report; default to token-weighted for "cost per delivered token."
- **A ratio (e.g. comm fraction `comm/(comm+comp)`) is scale-free in VALUE** (the `n` cancels) **but its DISTRIBUTION still depends on weighting.** If you claim per-forward and per-token coincide, *show both* (overlaid/2-row) and say *why* (e.g. fraction ⊥ tokens-committed because comm/comp are set by batch size) — don't assert it.
- **Report distributions, not point estimates.** These quantities are right-skewed (1-token forwards, desync tail) → histogram + **both mean and median**. A "typical..mean range" is a poor substitute for a histogram; the reviewer will (rightly) ask for the distribution.
- **When an identity must hold, compute both sides at the same granularity.** e.g. `intrinsic_s_k + straggler_waste = batch_S_k` only if all three are block-weighted (repeat the call's `S_k` per block); mixing call-weighted and block-weighted breaks the identity and the figure.

## 7. Figures are first-class — a `plot_<id>.py`, a `figures/`, and a stats JSON

For any distribution or sweep result, **plot it**; tables and "x..y" ranges do not convey a distribution. Add `plot_<id>.py` beside `parse_<id>.py`.
- **No-GPU, re-runnable:** `plot_<id>.py` reads the **saved** CSVs / `.sqlite` (never re-runs the model) and writes PNGs to the **repo** `experiments/<…>/<exp>/figures/` (small, versioned, referenced by the README with relative `figures/foo.png`). Pick one figures dir and use it everywhere — don't mix `figures/` and `assets/`.
- **Single source of truth:** have `plot_<id>.py` (or the parser) also dump the headline scalars to a `*_dist_stats.json`; cite those in the report so table/figure/text can't drift. After editing numbers, re-grep the README for stale values and reconcile rounding against the figure's own formatting.
- **For distributions:** histogram + mean (dashed) + median (dotted) annotated; label whether it is per-forward or per-token (and the weighting); note clipped tails (`N% > xmax, tail to …`) instead of letting a clip pile-up masquerade as a mode.
- `matplotlib` may be absent in the env — `pip install matplotlib` once; use `matplotlib.use("Agg")`.

## 8. Workload must be representative and GPU-bound

A profiling number is only as good as its operating point.
- **Drive a real dataset at sustained concurrency**, not a handful of hand-picked prompts in burst `curl`s. Use a load driver that keeps exactly `concurrency` requests in flight (cycling a real prompt set, e.g. HumanEval) so the running batch stays full; bound the nsys capture window to the steady state.
- **Realized batch ≠ nominal concurrency.** Requests desync and finish staggered; log and report the realized batch-size mix (the counter's `batch_size` per forward), and confirm the capture is actually GPU-bound (dominant batch ≈ concurrency for a large fraction of forwards). At bs≈1 the dLLM loop is host-bound and every per-token/comm number is unrepresentative.
- Issue more requests than the concurrency (e.g. `~4×`) so the batch is near-full for most of a short, trace-bounded capture rather than mostly ramp/drain.

## 9. Process tips (cheap insurance against expensive re-runs)

- **The data is method-independent; the parser is cheap.** Collect raw traces/CSVs once, then iterate parsing/plotting offline — never re-run the GPU job to fix an analysis bug.
- **When a reviewer questions a method, MEASURE the answer, don't argue it** (e.g. per-replay CV to test "is one number valid"; per-forward-vs-per-token histograms to test a weighting claim). Bake the validation into the parser/figure so it's reproducible.
- **Lead with the metric plan when asked (or when results get confusing):** a short table of *metric · level · unit · source · what-it-answers* before any numbers orients the reader; structure the Results section to mirror it.
- **Flag superseded results loudly** (a blockquote at the top) when a re-run corrects an earlier number, and say exactly what was wrong and why.
- `nsys` may not be on `PATH` in a non-interactive/`nohup` shell even if it works interactively — verify (`command -v nsys`) before launching a long background capture.

## 10. Checklist before calling an experiment done

- [ ] Scripts in `experiments/<family>/<subject>/<exp>/`; data under the mirrored `$DATA_ROOT/...` (nothing important in `/tmp` or `/root`).
- [ ] `run_*.sh` is a single copy-pasteable entry point; records commit/model/GPU+interconnect/shape/concurrency.
- [ ] Workload is a real dataset at sustained, GPU-bound concurrency; realized batch-size mix logged and reported (not assumed = nominal concurrency).
- [ ] Report is the folder `README.md`, full (all template sections), not a `notes/` stub.
- [ ] Every number labels its metric and states the operating point; comm numbers come from graph-ON node-trace, not eager.
- [ ] Per-token vs per-forward stated explicitly (with weighting); distributions shown as histograms with **mean and median**, not a single number or a range; "per-forward = per-token" never asserted.
- [ ] Figures generated by a no-GPU `plot_*.py` into `figures/`, referenced with relative paths; scalars sourced from one `*_dist_stats.json` so text/table/figure agree (re-grep for stale numbers).
- [ ] Formatting: single-line bullets/paragraphs, ≤1 consecutive blank line (the `awk` check prints `1`).
- [ ] Plan and any prior/compared report are cross-linked; family README index updated; superseded results flagged.
