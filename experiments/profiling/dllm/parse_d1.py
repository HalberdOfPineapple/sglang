#!/usr/bin/env python3
"""Summarize D1 nsys CSVs: comm-vs-compute split + per-step NVTX phase cost.

Usage: parse_d1.py <REP>   where <REP> is the nsys output prefix, so that
  <REP>_cuda_gpu_kern_sum.csv and <REP>_nvtx_pushpop_sum.csv exist.

Kernel names contain commas, so parse with csv.DictReader (never naive split).
See notes/experiment_20260619_d1_comm_decomposition.md for interpretation/caveats.
"""
import csv
import re
import sys

COMM_RE = re.compile(
    r"nccl|all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|"
    r"cross_device|custom_all|one.?shot|nvshmem|deep.?ep|dispatch|combine|"
    r"sendrecv|broadcast",
    re.I,
)


def comm_op(name: str) -> str:
    n = name.lower()
    if "allreduce" in n or "all_reduce" in n:
        return "AllReduce(TP)"
    if "alltoall" in n or "all_to_all" in n:
        return "AllToAll(EP)"
    if "allgather" in n or "all_gather" in n:
        return "AllGather"
    if "reducescatter" in n or "reduce_scatter" in n:
        return "ReduceScatter"
    if "broadcast" in n:
        return "Broadcast"
    return "COMM-other"


def kern_summary(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    tot = sum(int(r["Total Time (ns)"]) for r in rows)
    if not tot:
        print("  (empty kernel summary)")
        return
    comm = [r for r in rows if COMM_RE.search(r["Name"])]
    comm_t = sum(int(r["Total Time (ns)"]) for r in comm)
    print(f"  Total GPU kernel time: {tot/1e6:.1f} ms  ({len(rows)} distinct kernels)")
    print(f"    COMM   : {comm_t/1e6:9.1f} ms ({100*comm_t/tot:5.1f}%)")
    print(f"    COMPUTE: {(tot-comm_t)/1e6:9.1f} ms ({100*(tot-comm_t)/tot:5.1f}%)")
    if comm:
        print("  Comm by op:")
        agg = {}
        for r in comm:
            op = comm_op(r["Name"])
            t, i = agg.get(op, (0, 0))
            agg[op] = (t + int(r["Total Time (ns)"]), i + int(r["Instances"]))
        for op, (t, i) in sorted(agg.items(), key=lambda x: -x[1][0]):
            print(f"    {op:16s} {t/1e6:9.1f} ms  {i:6d} inst  ({100*t/tot:4.1f}% of GPU)")
    print("  Top kernels:")
    for r in sorted(rows, key=lambda r: -int(r["Total Time (ns)"]))[:8]:
        tag = "COMM" if COMM_RE.search(r["Name"]) else "COMP"
        nm = r["Name"].split("(")[0].strip().strip('"')[:58]
        print(f"    {int(r['Total Time (ns)'])/1e6:8.1f} ms {int(r['Instances']):6d} [{tag}] {nm}")


PHASES = ["dllm_forward.step", "dllm_select.step", "dllm_final_forward", "dllm_prefill_forward"]


def nvtx_gpu_proj_summary(path: str) -> None:
    """GPU time projected onto NVTX ranges — the meaningful per-phase metric.

    `Total Proj Time` = GPU busy time attributed to the range; `Total Range Time`
    = CPU push/pop wall-time. The two diverge a lot because forward launches a
    CUDA graph async (CPU returns early) and the host selection blocks on `.item()`
    — so compare GPU-proj across phases, and read CPU-range as wall-clock/host cost.
    """
    rows = list(csv.DictReader(open(path)))

    def agg(prefix):
        sel = [r for r in rows if r["Range"].split(":")[-1].startswith(prefix)]
        if not sel:
            return None
        gpu = sum(float(r["Total Proj Time (ns)"]) for r in sel)
        cpu = sum(float(r["Total Range Time (ns)"]) for r in sel)
        i = sum(int(r["Range Instances"]) for r in sel)
        return gpu, cpu, i

    print("  Per-phase (GPU-projected vs CPU-range), per call:")
    print(f"    {'phase':22s} {'GPU(ms)':>9} {'CPUrange(ms)':>13} {'inst':>5}")
    for p in PHASES:
        r = agg(p)
        if not r:
            continue
        gpu, cpu, i = r
        print(f"    {p:22s} {gpu/i/1e6:9.2f} {cpu/i/1e6:13.2f} {i:5d}")
    f, s = agg("dllm_forward.step"), agg("dllm_select.step")
    if f and s:
        steps = f[2]
        tot = (f[0] + s[0]) / steps / 1e6
        print(f"    per-step GPU total (fwd+sel): {tot:.2f} ms  "
              f"(forward {100*f[0]/(f[0]+s[0]):.0f}% / select {100*s[0]/(f[0]+s[0]):.0f}%)")


def nvtx_pushpop_summary(path: str) -> None:
    rows = list(csv.DictReader(open(path)))

    def agg(prefix):
        sel = [r for r in rows if r["Range"].split(":")[-1].startswith(prefix)]
        if not sel:
            return
        t = sum(float(r["Total Time (ns)"]) for r in sel)
        i = sum(int(r["Instances"]) for r in sel)
        print(f"    {prefix:22s} CPUrange sum={t/1e6:8.1f} ms  inst={i:5d}  avg={t/i/1e3:8.1f} us/call")

    print("  Per-phase NVTX CPU push/pop (wall-time; read with care):")
    for p in PHASES:
        agg(p)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: parse_d1.py <REP-prefix>")
    rep = sys.argv[1]
    print(f"\n===== D1 summary for {rep} =====")
    print("[cuda_gpu_kern_sum]")
    kern_summary(f"{rep}_cuda_gpu_kern_sum.csv")
    print("[nvtx_gpu_proj_sum]  <-- use this for per-phase GPU cost")
    try:
        nvtx_gpu_proj_summary(f"{rep}_nvtx_gpu_proj_sum.csv")
    except FileNotFoundError:
        print("  (run nsys stats --report nvtx_gpu_proj_sum first)")
    print("[nvtx_pushpop_sum]")
    nvtx_pushpop_summary(f"{rep}_nvtx_pushpop_sum.csv")


if __name__ == "__main__":
    main()
