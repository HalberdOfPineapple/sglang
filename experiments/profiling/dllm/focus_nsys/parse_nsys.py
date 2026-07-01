#!/usr/bin/env python3
"""Parse F6 nsys stats CSVs into a kernel-level bottleneck report.

For each captured tag ({focus_kernel,lowconf}_c{conc}) reads the nsys-stats CSVs
(produced by `nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum,cuda_gpu_trace`):

  1. GPU-busy vs capture-span (from *_cuda_gpu_trace.csv): sum of kernel durations
     / (max end − min start). The residual is GPU-idle = exposed host / launch-gap
     time — the key "eager launch-bound" evidence.
  2. Kernel-time by CATEGORY (from *_cuda_gpu_kern_sum.csv): MoE/GEMM, attention,
     moe-route/topk, norm/elementwise, embed/lm_head, other — where the GPU time
     actually goes.
  3. Per-NVTX-phase GPU-projected time (from *_nvtx_gpu_proj_sum.csv): FOCUS
     phases (focus_prefix / focus_l1_attn / focus_suffix / commit / final_forward)
     vs LowConfidence (dllm_forward / dllm_select).

Usage: parse_nsys.py PROFILES_DIR
"""
import csv
import glob
import os
import re
import sys


def _find(prof, tag, report):
    hits = glob.glob(os.path.join(prof, f"{tag}*_{report}.csv"))
    return hits[0] if hits else None


def _rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _col(row, *cands):
    """Return the first present column (case/space-insensitive contains match)."""
    keys = {k.lower().strip(): k for k in row.keys()}
    for c in cands:
        for lk, orig in keys.items():
            if c in lk:
                return row[orig]
    return None


# kernel-name -> coarse category
def _category(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("nccl",)):
        return "comm"
    if any(s in n for s in ("moe", "grouped", "group_gemm", "groupgemm", "fused_moe", "silu", "gate_up", "w13", "w2")):
        return "moe/gemm"
    if any(s in n for s in ("gemm", "cutlass", "matmul", "linear", "sm80", "s16816", "gett", "ampere")):
        return "moe/gemm"
    if any(s in n for s in ("attn", "attention", "prefill", "decode", "flashinfer", "paged", "bmm", "rope")):
        return "attention"
    if any(s in n for s in ("topk", "argmax", "sort", "routing", "router", "moe_align", "expert")):
        return "moe-route/topk"
    if any(s in n for s in ("norm", "rms", "layernorm", "add", "mul", "elementwise", "cast", "copy", "index", "scatter", "gather", "softmax", "activation")):
        return "norm/elementwise"
    if any(s in n for s in ("embed", "lm_head", "logit", "vocab")):
        return "embed/lm_head"
    return "other"


def report_trace(prof, tag):
    path = _find(prof, tag, "cuda_gpu_trace")
    rows = _rows(path)
    if not rows:
        return None
    starts, ends, durs = [], [], []
    for r in rows:
        st = _num(_col(r, "start"))
        du = _num(_col(r, "duration", "dur"))
        if du <= 0:
            continue
        starts.append(st)
        ends.append(st + du)
        durs.append(du)
    if not durs:
        return None
    span = max(ends) - min(starts)
    busy = sum(durs)
    return {
        "n_kernels": len(durs),
        "span_ms": span / 1e6,
        "busy_ms": busy / 1e6,
        "busy_pct": 100.0 * busy / span if span else 0.0,
        "idle_pct": 100.0 * (1 - busy / span) if span else 0.0,
        "avg_kernel_us": (busy / len(durs)) / 1e3,
    }


def report_kern(prof, tag, topn=12):
    path = _find(prof, tag, "cuda_gpu_kern_sum")
    rows = _rows(path)
    if not rows:
        return None, None
    tot = 0.0
    cats = {}
    items = []
    for r in rows:
        t = _num(_col(r, "total time", "total_time"))
        name = _col(r, "name", "kernel") or "?"
        inst = _num(_col(r, "instances", "count", "num"))
        tot += t
        c = _category(name)
        cats[c] = cats.get(c, 0.0) + t
        items.append((t, inst, name))
    items.sort(reverse=True)
    cat_sorted = sorted(cats.items(), key=lambda x: -x[1])
    return (
        {"total_ms": tot / 1e6,
         "cats": [(c, v / 1e6, 100 * v / tot if tot else 0) for c, v in cat_sorted],
         "top": [(t / 1e6, int(i), n) for t, i, n in items[:topn]]},
        tot,
    )


_PHASE_MAP = [
    ("focus_prefix", "P: focus_prefix (L0+L1qkv, full B)"),
    ("focus_l1_attn", "A1: focus_l1_attn (L1 attn on |S|)"),
    ("focus_suffix", "S: focus_suffix (L2..L on |S|)"),
    ("dllm_focus_commit", "commit"),
    ("dllm_focus_final_forward", "final full forward (KV repop)"),
    ("dllm_prefill_forward", "prefill (mask-free)"),
    ("dllm_forward", "LC: dllm_forward.step"),
    ("dllm_select", "LC: dllm_select.step"),
]


def report_nvtx(prof, tag):
    path = _find(prof, tag, "nvtx_gpu_proj_sum")
    rows = _rows(path)
    if not rows:
        return None
    agg = {}
    for r in rows:
        rng = (_col(r, "range", "name") or "").strip().lstrip(":")
        t = _num(_col(r, "proj time", "total time", "total_time"))
        # bucket by known phase prefixes (step-numbered ranges collapse together)
        label = None
        for pref, lab in _PHASE_MAP:
            if rng.startswith(pref):
                label = lab
                break
        if label is None:
            continue
        agg[label] = agg.get(label, 0.0) + t
    if not agg:
        return None
    tot = sum(agg.values())
    return sorted(
        [(lab, ms / 1e6, 100 * ms / tot if tot else 0) for lab, ms in agg.items()],
        key=lambda x: -x[1],
    )


def main():
    prof = sys.argv[1] if len(sys.argv) > 1 else "."
    tags = sorted(
        {
            re.match(r"(.+?)_cuda_gpu_kern_sum\.csv", os.path.basename(p)).group(1)
            for p in glob.glob(os.path.join(prof, "*_cuda_gpu_kern_sum.csv"))
        }
    )
    if not tags:
        print(f"No *_cuda_gpu_kern_sum.csv in {prof}")
        return
    print("=" * 78)
    print("F6 — nsys kernel-level bottleneck decomposition (LLaDA2.0-mini 1xA100 eager)")
    print("=" * 78)
    for tag in tags:
        print(f"\n### {tag}")
        tr = report_trace(prof, tag)
        if tr:
            print(f"  GPU busy {tr['busy_ms']:.1f}ms / span {tr['span_ms']:.1f}ms "
                  f"= {tr['busy_pct']:.1f}% busy, {tr['idle_pct']:.1f}% IDLE "
                  f"(exposed host/launch-gap) | {tr['n_kernels']} kernels, "
                  f"avg {tr['avg_kernel_us']:.1f}us/kernel")
        kern, _ = report_kern(prof, tag)
        if kern:
            print(f"  kernel GPU time by category (Σ={kern['total_ms']:.1f}ms):")
            for c, ms, pc in kern["cats"]:
                print(f"      {c:>18}: {ms:8.1f}ms {pc:5.1f}%")
            print("  top kernels:")
            for ms, inst, name in kern["top"]:
                print(f"      {ms:7.1f}ms  x{inst:<6} {name[:60]}")
        nv = report_nvtx(prof, tag)
        if nv:
            print("  per-NVTX-phase GPU-projected time:")
            for lab, ms, pc in nv:
                print(f"      {lab:<38}: {ms:8.1f}ms {pc:5.1f}%")
    print()


if __name__ == "__main__":
    main()
