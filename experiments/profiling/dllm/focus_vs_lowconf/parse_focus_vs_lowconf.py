#!/usr/bin/env python3
"""Aggregate FOCUS-vs-LowConfidence results into a comparison table + stats JSON.

Reads ``$LOGS/{focus,lowconfidence}_c<conc>_result.json`` (driver output) and the
FOCUS redundancy logs ``focus_c<conc>_redundancy.log`` (lines of
``[focus] |S|/(B*bs) = k/n = r ...``), and emits:
  - a per-concurrency comparison table (tok/s, latency, FOCUS speedup, redundancy),
  - ``focus_vs_lowconf_dist_stats.json`` (single source of truth for the report).

Usage: parse_focus_vs_lowconf.py LOGS_DIR
"""
import csv
import glob
import json
import os
import re
import statistics
import sys


def _load_result(logs, algo, conc):
    path = os.path.join(logs, f"{algo}_c{conc}_result.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["summary"]


def _load_redundancy(logs, conc):
    """Return list of per-step Sigma|S|/(B*bs) ratios for FOCUS at this conc."""
    path = os.path.join(logs, f"focus_c{conc}_redundancy.csv")
    ratios = []
    if not os.path.exists(path):
        return ratios
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ratios.append(float(row["ratio"]))
            except (KeyError, ValueError):
                continue
    return ratios


def main():
    logs = sys.argv[1]
    concs = sorted(
        {
            int(re.search(r"_c(\d+)_result", os.path.basename(p)).group(1))
            for p in glob.glob(os.path.join(logs, "*_c*_result.json"))
        }
    )
    stats = {"by_conc": {}}
    rows = []
    for c in concs:
        lc = _load_result(logs, "lowconfidence", c)
        fo = _load_result(logs, "focus", c)
        red = _load_redundancy(logs, c)
        red_mean = statistics.mean(red) if red else None
        red_median = statistics.median(red) if red else None
        speedup = (fo["tok_s"] / lc["tok_s"]) if (lc and fo and lc["tok_s"]) else None
        lat_ratio = (
            (fo["lat_mean_s"] / lc["lat_mean_s"])
            if (lc and fo and lc["lat_mean_s"])
            else None
        )
        stats["by_conc"][c] = {
            "lowconf_tok_s": lc["tok_s"] if lc else None,
            "focus_tok_s": fo["tok_s"] if fo else None,
            "speedup": speedup,
            "lowconf_lat_mean_s": lc["lat_mean_s"] if lc else None,
            "focus_lat_mean_s": fo["lat_mean_s"] if fo else None,
            "lat_ratio_focus_over_lowconf": lat_ratio,
            "focus_redundancy_mean": red_mean,
            "focus_redundancy_median": red_median,
            "focus_redundancy_n": len(red),
        }
        rows.append(
            (
                c,
                lc["tok_s"] if lc else float("nan"),
                fo["tok_s"] if fo else float("nan"),
                speedup if speedup else float("nan"),
                lc["lat_mean_s"] if lc else float("nan"),
                fo["lat_mean_s"] if fo else float("nan"),
                red_mean if red_mean else float("nan"),
            )
        )

    out_json = os.path.join(logs, "focus_vs_lowconf_dist_stats.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== FOCUS vs LowConfidence (LLaDA2.0-mini, 1xA100, TP=1, eager) ===")
    print(
        f"{'conc':>4} | {'LowConf tok/s':>13} | {'FOCUS tok/s':>11} | "
        f"{'speedup':>7} | {'LC lat(s)':>9} | {'FO lat(s)':>9} | {'FO redund':>9}"
    )
    print("-" * 80)
    for (c, lct, fot, sp, lcl, fol, rm) in rows:
        print(
            f"{c:>4} | {lct:>13.0f} | {fot:>11.0f} | {sp:>7.2f} | "
            f"{lcl:>9.1f} | {fol:>9.1f} | {rm:>9.3f}"
        )
    print(f"\nstats -> {out_json}")


if __name__ == "__main__":
    main()
