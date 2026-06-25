# Experiments

Self-contained experiments for the dLLM-on-SGLang research (see top-level
`CLAUDE.md` / `AGENTS.md`). **Scripts live here in the repo; large outputs
(profiles, logs, CSVs) are data and go to `/cephfs`** in a tree that *mirrors*
this `experiments/` hierarchy, so scripts and their data line up 1:1.

## Conventions

- **One folder per experiment.** Each leaf folder is self-contained and holds the
  full report: its driver script(s), parser(s), and a `README.md` that *is* the
  experiment report (Summary / Setup / Method / Results / Caveats / Next),
  following `notes/templates/experiment_report_template.md`. The report lives here
  in the subfolder, not split into `notes/`.
- **Data root mirrors the repo.** Default `DATA_ROOT=/cephfs/shared/wxli/sglang-dllm`;
  a leaf at `experiments/<family>/<subject>/<exp>/` writes data to
  `$DATA_ROOT/<family>/<subject>/<exp>/{profiles,logs}/`. Never `/tmp` or `/root`.
- **Reproducibility.** Each run records git commit, model, GPU count/type +
  interconnect, TP/EP/DP shape, `mem-fraction-static`, concurrency, prompts.

## Tree

```
experiments/
  profiling/
    dllm/                              # dLLM distributed-efficiency profiling plan
      README.md                        # family overview, maps the D1..D9 plan to folders
      common/                          # shared helpers (if any)
      d1_comm_decomposition/           # D1 — per-step comm decomposition & exposed fraction
        run_d1.sh  parse_d1.py  README.md
      # d2.._d9 ...                    # added as the plan progresses
```

Plan of record: `notes/dllm_baseline_profiling_plan.md` (experiments D1–D9).
