---
name: experiment-report
description: Lay out an experiment under experiments/, route scripts (repo) vs data (CephFS mirror) correctly, and write a dense dLLM/SGLang experiment report inside the experiment folder. Use when creating/organizing a profiling or benchmark experiment, storing its scripts/outputs, or writing/updating its report.
---

# Experiment Layout, Storage, and Report Writing

Conventions for this repo's dLLM-on-SGLang experiments. Two inseparable concerns: **where things live** (scripts in repo, data on CephFS, report in the experiment folder) and **how the report reads** (dense, Typora-friendly, metric-disciplined). Canonical examples to copy from: `notes/templates/experiment_report_template.md`, the full report `experiments/profiling/dllm/d1_comm_decomposition/README.md`, its scripts `run_d1.sh`/`parse_d1.py`, and the legacy `notes/experiment_20260619_d1_comm_decomposition.md`.

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

## 6. Checklist before calling an experiment done

- [ ] Scripts in `experiments/<family>/<subject>/<exp>/`; data under the mirrored `$DATA_ROOT/...` (nothing important in `/tmp` or `/root`).
- [ ] `run_*.sh` is a single copy-pasteable entry point; records commit/model/GPU+interconnect/shape/concurrency.
- [ ] Report is the folder `README.md`, full (all template sections), not a `notes/` stub.
- [ ] Every number labels its metric and states the operating point; comm numbers come from graph-ON node-trace, not eager.
- [ ] Formatting: single-line bullets/paragraphs, ≤1 consecutive blank line (the `awk` check prints `1`).
- [ ] Plan and any prior/compared report are cross-linked; family README index updated.
