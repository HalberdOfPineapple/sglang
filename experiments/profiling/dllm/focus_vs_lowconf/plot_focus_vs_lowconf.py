#!/usr/bin/env python3
"""Figures for the FOCUS-vs-LowConfidence experiment (no GPU; reads saved data).

Reads the driver result JSONs + FOCUS redundancy logs under LOGS_DIR and writes
PNGs into the repo ``figures/`` dir:
  - fig1_throughput.png   : tok/s, FOCUS vs LowConfidence, per concurrency.
  - fig2_redundancy_hist.png : per-reduced-step Sigma|S|/(B*bs) distribution (FOCUS).

Usage: plot_focus_vs_lowconf.py LOGS_DIR [FIGURES_DIR]
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def _result(logs, algo, c):
    p = os.path.join(logs, f"{algo}_c{c}_result.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def _redundancy(logs, c):
    p = os.path.join(logs, f"focus_c{c}_redundancy.csv")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    out.append(float(row["ratio"]))
                except (KeyError, ValueError):
                    continue
    return out


def main():
    logs = sys.argv[1]
    figs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
    os.makedirs(figs, exist_ok=True)
    concs = sorted(
        {
            int(re.search(r"_c(\d+)_result", os.path.basename(p)).group(1))
            for p in glob.glob(os.path.join(logs, "*_c*_result.json"))
        }
    )

    # Fig 1: throughput bars.
    lc = [(_result(logs, "lowconfidence", c) or {}).get("tok_s", 0) for c in concs]
    fo = [(_result(logs, "focus", c) or {}).get("tok_s", 0) for c in concs]
    x = range(len(concs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], lc, w, label="LowConfidence", color="#888")
    ax.bar([i + w / 2 for i in x], fo, w, label="FOCUS (a=1.5)", color="#2a7")
    for i, (l, f) in enumerate(zip(lc, fo)):
        if l:
            ax.text(i + w / 2, f, f"{f/l:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in concs])
    ax.set_xlabel("concurrency (= max_running_requests)")
    ax.set_ylabel("output tok/s")
    ax.set_title("FOCUS vs LowConfidence throughput (LLaDA2.0-mini, 1xA100, eager)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig1_throughput.png"), dpi=120)

    # Fig 2: redundancy histogram (pool all concurrencies, annotate per-conc means).
    all_red = []
    for c in concs:
        all_red += _redundancy(logs, c)
    if all_red:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_red, bins=20, range=(0, 1), color="#2a7", alpha=0.8)
        m, md = statistics.mean(all_red), statistics.median(all_red)
        ax.axvline(m, color="k", ls="--", label=f"mean {m:.3f}")
        ax.axvline(md, color="k", ls=":", label=f"median {md:.3f}")
        ax.set_xlabel("per-reduced-step  Sigma|S| / (B * bs)   (1.0 = no eviction)")
        ax.set_ylabel("# reduced steps")
        ax.set_title("FOCUS processed-token fraction per step (lower = more eviction)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "fig2_redundancy_hist.png"), dpi=120)
        print(f"redundancy: n={len(all_red)} mean={m:.3f} median={md:.3f}")

    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
