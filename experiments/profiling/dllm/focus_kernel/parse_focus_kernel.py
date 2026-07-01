#!/usr/bin/env python3
"""Parse the FOCUS kernel ON/OFF profiling runs.

Reads from LOGS:
  * k{0,1}_c{conc}_T_result.json      — driver throughput/latency
  * k{0,1}_c{conc}_T_redundancy.csv   — per-step Σ|S|/(B·bs) (confirms identical eviction)
  * k{0,1}_c{PHASE}_P_server.log      — aggregated `[focus-timing]` phase shares

Emits a compact table: tok/s (kernel OFF vs ON, speedup), the redundancy means
(must match — the kernels don't change *which* tokens are evicted, only how the
importance/selection are computed), and the per-phase ms/share split at the phase
concurrency (does the `select` + importance-bearing host cost shrink?).

Usage: parse_focus_kernel.py LOGS_DIR
"""
import glob
import json
import os
import re
import sys


def _read_result(logs, tag):
    p = os.path.join(logs, f"{tag}_result.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["summary"]


def _read_redundancy(logs, tag):
    p = os.path.join(logs, f"{tag}_redundancy.csv")
    if not os.path.exists(p):
        return None
    ratios = []
    with open(p) as f:
        next(f, None)  # header
        for line in f:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    ratios.append(float(parts[3]))
                except ValueError:
                    pass
    if not ratios:
        return None
    ratios.sort()
    n = len(ratios)
    return {
        "n": n,
        "mean": sum(ratios) / n,
        "median": ratios[n // 2],
        "min": ratios[0],
        "max": ratios[-1],
    }


_TIMING_RE = re.compile(r"\[focus-timing\] bs=(\d+) steps=(\d+) total=([\d.]+)ms :: (.*)")
_PART_RE = re.compile(r"(\w+)=([\d.]+)\((\d+)%\)")


def _read_phase_timing(logs, tag):
    """Aggregate all [focus-timing] lines in a server log into total ms per phase."""
    p = os.path.join(logs, f"{tag}_server.log")
    if not os.path.exists(p):
        return None
    phase_ms = {}
    grand_total = 0.0
    n_lines = 0
    with open(p, errors="ignore") as f:
        for line in f:
            m = _TIMING_RE.search(line)
            if not m:
                continue
            n_lines += 1
            for name, ms, _pct in _PART_RE.findall(m.group(4)):
                phase_ms[name] = phase_ms.get(name, 0.0) + float(ms)
                grand_total += float(ms)
    if n_lines == 0 or grand_total <= 0:
        return None
    shares = {k: (v, 100.0 * v / grand_total) for k, v in phase_ms.items()}
    return {"n_lines": n_lines, "total_ms": grand_total, "shares": shares}


def main():
    logs = sys.argv[1] if len(sys.argv) > 1 else "."
    # discover concurrencies from throughput result files
    concs = sorted(
        {
            int(re.search(r"_c(\d+)_T_result", os.path.basename(p)).group(1))
            for p in glob.glob(os.path.join(logs, "k*_c*_T_result.json"))
        }
    )
    print("=" * 74)
    print("FOCUS Phase-B kernel (SGLANG_FOCUS_KERNEL) ON vs OFF — LLaDA2.0-mini 1xA100")
    print("=" * 74)

    print("\n## Throughput (mode T, no phase-timing syncs)")
    print(f"{'conc':>4} | {'OFF tok/s':>9} {'ON tok/s':>9} {'speedup':>8} | "
          f"{'OFF lat':>7} {'ON lat':>7} | {'redund OFF/ON (mean)':>22}")
    print("-" * 74)
    for c in concs:
        off = _read_result(logs, f"k0_c{c}_T")
        on = _read_result(logs, f"k1_c{c}_T")
        if not off or not on:
            continue
        sp = on["tok_s"] / off["tok_s"] if off["tok_s"] else float("nan")
        rd_off = _read_redundancy(logs, f"k0_c{c}_T")
        rd_on = _read_redundancy(logs, f"k1_c{c}_T")
        rd_s = (
            f"{rd_off['mean']:.3f}/{rd_on['mean']:.3f}"
            if rd_off and rd_on else "n/a"
        )
        print(f"{c:>4} | {off['tok_s']:>9.0f} {on['tok_s']:>9.0f} {sp:>7.2f}x | "
              f"{off['lat_mean_s']:>6.2f}s {on['lat_mean_s']:>6.2f}s | {rd_s:>22}")

    print("\n## Per-phase split (mode P, phase-timing ON — shares are the signal,")
    print("## absolute ms inflated by boundary syncs). OFF vs ON at phase conc.")
    for tagp, label in (("P", ""),):
        # find phase conc
        pfiles = glob.glob(os.path.join(logs, f"k0_c*_{tagp}_server.log"))
        if not pfiles:
            continue
        pc = int(re.search(r"_c(\d+)_", os.path.basename(pfiles[0])).group(1))
        off = _read_phase_timing(logs, f"k0_c{pc}_{tagp}")
        on = _read_phase_timing(logs, f"k1_c{pc}_{tagp}")
        if not off or not on:
            print("  (phase-timing logs missing)")
            continue
        phases = sorted(
            set(off["shares"]) | set(on["shares"]),
            key=lambda k: -off["shares"].get(k, (0, 0))[1],
        )
        print(f"\n  conc={pc}  (OFF total={off['total_ms']:.0f}ms / {off['n_lines']} blocks, "
              f"ON total={on['total_ms']:.0f}ms / {on['n_lines']} blocks)")
        print(f"  {'phase':>8} | {'OFF ms':>9} {'OFF %':>6} | {'ON ms':>9} {'ON %':>6}")
        print("  " + "-" * 48)
        for ph in phases:
            o_ms, o_pc = off["shares"].get(ph, (0.0, 0.0))
            n_ms, n_pc = on["shares"].get(ph, (0.0, 0.0))
            print(f"  {ph:>8} | {o_ms:>9.0f} {o_pc:>5.1f}% | {n_ms:>9.0f} {n_pc:>5.1f}%")
    print()


if __name__ == "__main__":
    main()
