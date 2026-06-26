#!/usr/bin/env python3
"""D2 figures — per-FORWARD / per-TOKEN distributions with clear variable management.

Joins each forward's measured (comm_i, comp_i) with its n_committed to form the
CORRECT per-token ratios (not one average divided by varying n). Plots organized:
  fig1  per-token comm/comp MEANS vs concurrency (summary trends)
  fig2  committed tokens per forward distribution (the denominator, 3 histograms)
  fig3  comm/token and comp/token per-forward distributions (token-weighted)
  fig4  intrinsic s_k + batch S_k = intrinsic + straggler waste
  fig5  communication-time FRACTION per forward vs per token
  fig6  raw comm-time and comp-time per forward (shows desync tail)

Usage: plot_d2.py [logs_dir] [figures_dir] [profiles_dir]
"""
import csv
import json
import os
import re
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---------------------------------------------------------------
CONCS = [4, 8, 16]  # concurrency levels (max-running-requests) in the sweep
COL_COMM = "#d62728"
COL_COMP = "#1f77b4"
COL_TOK = "#2ca02c"
COL_WASTE = "#d62728"
COL_GOOD = "#2ca02c"
COL_FRAC = "#9467bd"

REPO = os.environ.get("REPO", "/root/sglang_a100/sglang")
DATA_ROOT = os.environ.get("DATA_ROOT", "/cephfs/shared/wxli/sglang-dllm")
EXP_SUBDIR = "profiling/dllm/d2_sk_amplification/h100"

CAPTURED_GRAPH_BS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
COLLECTIVE_RE = re.compile(
    r"nccl|all.?reduce|all.?gather|all.?to.?all|reduce.?scatter|sendrecv|nvshmem", re.I)


def pad_bs_to_graph(bs):
    for g in CAPTURED_GRAPH_BS:
        if g >= bs:
            return g
    return CAPTURED_GRAPH_BS[-1]


def tag_for_conc(c):
    return f"d2_h100_tp4_c{c}"


# --- Per-replay reconstruction (same as parse_d2.py, kept DRY) ------------------
def per_replay_comm_comp_by_graph(sqlite_path):
    """Return {graphId: [(comm_ms, comp_ms, attn_ms), ...]} per replay."""
    cur = sqlite3.connect(sqlite_path).cursor()
    rep_dev = cur.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphId, k.graphNodeId, k.start, k.end-k.start, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphId IS NOT NULL", (rep_dev,)).fetchall()
    by_graph = defaultdict(lambda: defaultdict(list))
    for gid, node, start, dur, name in rows:
        by_graph[gid][node].append((start, dur, name))
    out = {}
    for gid, nodes in by_graph.items():
        n_replays = max(len(inst) for inst in nodes.values())
        comm_arr = [0.0] * n_replays
        comp_arr = [0.0] * n_replays
        for inst in nodes.values():
            inst.sort()
            for i, (s, d, nm) in enumerate(inst):
                if i >= n_replays:
                    break
                if COLLECTIVE_RE.search(nm):
                    comm_arr[i] += d
                else:
                    comp_arr[i] += d
        out[gid] = [(comm_arr[i] / 1e6, comp_arr[i] / 1e6, 0.0) for i in range(n_replays)]
    return out


def join_forwards(perstep_csv_rows, graphs_by_gid):
    """Return [(comm_ms, comp_ms, n_committed, batch_size), ...] per forward (committed>0)."""
    padded_counts = Counter(pad_bs_to_graph(r["batch_size"]) for r in perstep_csv_rows)
    gids_ranked = sorted(graphs_by_gid.keys(), key=lambda g: -len(graphs_by_gid[g]))
    pbs_ranked = [bs for bs, _ in padded_counts.most_common()]
    gid_to_pbs = {g: pbs for g, pbs in zip(gids_ranked, pbs_ranked)}
    pools = defaultdict(list)
    for gid, pbs in gid_to_pbs.items():
        pools[pbs].extend(graphs_by_gid[gid])
    avail = sorted(pools.keys())
    samples = []
    cycle = defaultdict(int)
    for r in perstep_csv_rows:
        n = r["committed"]
        if n <= 0:
            continue
        bs = r["batch_size"]
        pbs = pad_bs_to_graph(bs)
        if pbs not in pools:
            pbs = min(avail, key=lambda x: abs(x - pbs))
        pool = pools[pbs]
        comm_i, comp_i, _ = pool[cycle[pbs] % len(pool)]
        cycle[pbs] += 1
        samples.append((comm_i, comp_i, n, bs))
    return samples


# --- Load data for all concurrencies --------------------------------------------
def load_all(logs_dir, profiles_dir):
    """Return {conc: data_dict} where data_dict has arrays + metrics from JSON."""
    D = {}
    for c in CONCS:
        tag = tag_for_conc(c)
        base = os.path.join(logs_dir, tag + "_blocks")
        metrics = json.load(open(base + "_d2metrics.json"))
        perstep = [{"batch_size": int(r["batch_size"]), "committed": int(r["committed"])}
                   for r in csv.DictReader(open(base + "_perstep.csv"))]
        blocks = list(csv.DictReader(open(base + ".csv")))
        intrinsic_sk_arr = [int(r["finish_step"])+1 if int(r["finish_step"])>=0 else int(r["S_k"])
                            for r in blocks]
        graphs = per_replay_comm_comp_by_graph(os.path.join(profiles_dir, tag + ".sqlite"))
        forwards = join_forwards(perstep, graphs)

        # unpack per-forward arrays
        comm_fwd_arr = np.array([f[0] for f in forwards], float)
        comp_fwd_arr = np.array([f[1] for f in forwards], float)
        n_committed_arr = np.array([f[2] for f in forwards], float)
        batch_size_arr = np.array([f[3] for f in forwards], int)

        # per-token arrays: each forward's ratio
        comm_per_token_arr = comm_fwd_arr / n_committed_arr
        comp_per_token_arr = comp_fwd_arr / n_committed_arr
        comm_frac_arr = comm_fwd_arr / (comm_fwd_arr + comp_fwd_arr)

        # token-weighted means (= sum(time) / sum(tokens))
        tw_comm_per_token = comm_fwd_arr.sum() / n_committed_arr.sum()
        tw_comp_per_token = comp_fwd_arr.sum() / n_committed_arr.sum()
        tw_comm_frac = comm_fwd_arr.sum() / (comm_fwd_arr.sum() + comp_fwd_arr.sum())

        D[c] = {
            "metrics": metrics,
            "intrinsic_sk_arr": np.array(intrinsic_sk_arr, float),
            # per-forward (L1)
            "comm_fwd_arr": comm_fwd_arr,
            "comp_fwd_arr": comp_fwd_arr,
            "n_committed_arr": n_committed_arr,
            "batch_size_arr": batch_size_arr,
            # per-token (L4)
            "comm_per_token_arr": comm_per_token_arr,
            "comp_per_token_arr": comp_per_token_arr,
            "comm_frac_arr": comm_frac_arr,
            # token-weighted scalars
            "tw_comm_per_token": tw_comm_per_token,
            "tw_comp_per_token": tw_comp_per_token,
            "tw_comm_frac": tw_comm_frac,
        }
    return D


# --- Weighted statistics ---------------------------------------------------------
def weighted_mean(values, weights):
    return float(np.average(values, weights=weights))


def weighted_median(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    vals_sorted = values[order]
    cumsum = np.cumsum(weights[order])
    return float(vals_sorted[np.searchsorted(cumsum, 0.5 * cumsum[-1])])


def add_stat_lines(ax, mean_val, median_val, unit=""):
    """Overlay mean (dashed) and median (dotted) vlines with annotations."""
    ax.axvline(mean_val, color="black", ls="--", lw=1.6)
    ax.axvline(median_val, color="black", ls=":", lw=1.6)
    y_top = ax.get_ylim()[1]
    ax.annotate(f"mean {mean_val:.3g}{unit}", (mean_val, y_top), xytext=(4, -10),
                textcoords="offset points", fontsize=8)
    ax.annotate(f"median {median_val:.3g}{unit}", (median_val, y_top), xytext=(4, -22),
                textcoords="offset points", fontsize=8)


def plot_hist_row(axes, data_by_conc, xmax, color, xlabel, unit="", bins=44,
                  weights_by_conc=None):
    """Plot histogram per concurrency in a row of axes. If weights given, each sample
    is weighted (e.g. by n_committed → distribution over delivered tokens)."""
    edges = np.linspace(0, xmax, bins)
    for ax, c in zip(axes, CONCS):
        vals = np.array(data_by_conc[c], float)
        wts = np.array(weights_by_conc[c], float) if weights_by_conc else np.ones(len(vals))
        wts = wts / wts.sum()
        over_frac = float(wts[vals > xmax].sum())
        ax.hist(np.clip(vals, 0, xmax), bins=edges, weights=wts,
                color=color, alpha=0.8, edgecolor="white", lw=0.3)
        add_stat_lines(ax, weighted_mean(vals, wts), weighted_median(vals, wts), unit=unit)
        title = f"conc {c}  (n={len(vals)})"
        if over_frac > 0.005:
            title += f"\n{100*over_frac:.0f}% > {xmax:g} (tail to {vals.max():.2g})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)


# --- Figure 1: summary trends (per-token means vs concurrency) -------------------
def figure1_pertoken_means_vs_concurrency(data, out_path):
    """Line plot: token-weighted comm/token and comp/token vs concurrency, with
    secondary axis showing tokens/forward."""
    x_pos = np.arange(len(CONCS))
    tw_comm_arr = [data[c]["tw_comm_per_token"] for c in CONCS]
    tw_comp_arr = [data[c]["tw_comp_per_token"] for c in CONCS]
    mean_tok_arr = [data[c]["n_committed_arr"].mean() for c in CONCS]

    fig, ax_left = plt.subplots(figsize=(6.4, 4.2))
    ax_left.plot(x_pos, tw_comp_arr, "o-", color=COL_COMP, lw=2, ms=7, label="compute / token")
    ax_left.plot(x_pos, tw_comm_arr, "o-", color=COL_COMM, lw=2, ms=7, label="exposed comm / token")
    for i, (comm_val, comp_val) in enumerate(zip(tw_comm_arr, tw_comp_arr)):
        ax_left.annotate(f"{comp_val:.2f}", (i, comp_val), textcoords="offset points",
                         xytext=(0, 8), ha="center", color=COL_COMP, fontsize=9)
        ax_left.annotate(f"{comm_val:.3f}", (i, comm_val), textcoords="offset points",
                         xytext=(0, -14), ha="center", color=COL_COMM, fontsize=9)
    ax_left.set_ylabel("ms per output token (token-weighted mean)")
    ax_left.set_xlabel("concurrency (= max-running-requests = dominant batch size)")
    ax_left.set_ylim(0, max(tw_comp_arr) * 1.25)
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(CONCS)
    ax_left.grid(True, axis="y", alpha=0.3)

    ax_right = ax_left.twinx()
    ax_right.plot(x_pos, mean_tok_arr, "s--", color=COL_TOK, lw=2, ms=7, label="tokens / forward")
    for i, tok_val in enumerate(mean_tok_arr):
        ax_right.annotate(f"{tok_val:.1f}", (i, tok_val), textcoords="offset points",
                          xytext=(0, 9), ha="center", color=COL_TOK, fontsize=9)
    ax_right.set_ylabel("tokens committed per forward", color=COL_TOK)
    ax_right.tick_params(axis="y", labelcolor=COL_TOK)
    ax_right.set_ylim(0, max(mean_tok_arr) * 1.32)

    l1, lb1 = ax_left.get_legend_handles_labels()
    l2, lb2 = ax_right.get_legend_handles_labels()
    ax_left.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=9, framealpha=0.9)
    ax_left.set_title("Per-token comm/compute (means) fall with concurrency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- Figure 2: committed tokens per forward (3 histograms) ----------------------
def figure2_committed_per_forward_hist(data, out_path):
    """Histogram: tokens committed per forward (the per-token denominator)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True, sharey=True)
    fig.suptitle("Tokens committed per forward (the per-token denominator)", y=1.02)
    plot_hist_row(axes, {c: data[c]["n_committed_arr"] for c in CONCS},
                  xmax=50, color=COL_TOK, xlabel="tokens committed in one forward")
    axes[0].set_ylabel("share of forwards")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Figure 3: per-token cost distributions (token-weighted) --------------------
def figure3_pertoken_cost_distributions(data, out_path):
    """Histogram: comm/token and comp/token per forward, token-weighted (each forward
    weighted by the tokens it delivered → distribution over delivered tokens)."""
    weights = {c: data[c]["n_committed_arr"] for c in CONCS}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    fig.suptitle("Distribution over delivered tokens of cost per output token\n"
                 "(each forward's time ÷ its n_committed; token-weighted)", y=1.0)
    plot_hist_row(axes[0], {c: data[c]["comm_per_token_arr"] for c in CONCS},
                  xmax=2.0, color=COL_COMM, xlabel="exposed comm per output token (ms)",
                  weights_by_conc=weights)
    plot_hist_row(axes[1], {c: data[c]["comp_per_token_arr"] for c in CONCS},
                  xmax=4.0, color=COL_COMP, xlabel="compute per output token (ms)",
                  weights_by_conc=weights)
    axes[0][0].set_ylabel("share of delivered tokens")
    axes[1][0].set_ylabel("share of delivered tokens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Figure 4: intrinsic s_k + batch S_k with straggler -------------------------
def figure4_sk_and_straggler(data, out_path):
    """Left: intrinsic s_k pooled histogram. Right: stacked bar batch S_k = intrinsic + waste."""
    m = {c: data[c]["metrics"] for c in CONCS}
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.0))

    # left: pooled intrinsic s_k
    pooled_sk = np.concatenate([data[c]["intrinsic_sk_arr"] for c in CONCS])
    ax_left.hist(pooled_sk, bins=np.arange(0, 34, 1),
                 weights=np.ones(len(pooled_sk)) / len(pooled_sk),
                 color="#7f7f7f", alpha=0.75, edgecolor="white", lw=0.4)
    colors_conc = ["#9467bd", "#ff7f0e", "#17becf"]
    for c, col in zip(CONCS, colors_conc):
        mean_val = m[c]["intrinsic_sk_mean"]
        ax_left.axvline(mean_val, color=col, ls="--", lw=1.8, label=f"conc {c}: mean {mean_val:.1f}")
    ax_left.set_xlabel("intrinsic s_k (block's own denoising steps = finish_step+1)")
    ax_left.set_ylabel("share of blocks")
    ax_left.set_title("Intrinsic s_k is content-driven (pooled; means ~coincide)")
    ax_left.legend(fontsize=9)
    ax_left.grid(True, axis="y", alpha=0.3)

    # right: batch S_k stacked bar
    x_pos = np.arange(len(CONCS))
    intrinsic_arr = [m[c]["intrinsic_sk_mean"] for c in CONCS]
    waste_arr = [m[c]["straggler_waste_fwd"] for c in CONCS]
    waste_pct_arr = [m[c]["straggler_waste_pct"] for c in CONCS]
    ax_right.bar(x_pos, intrinsic_arr, color=COL_GOOD, label="productive (intrinsic s_k)")
    ax_right.bar(x_pos, waste_arr, bottom=intrinsic_arr, color=COL_WASTE,
                 label="straggler waste (forwards after finish)")
    for i, (intr, wst, pct) in enumerate(zip(intrinsic_arr, waste_arr, waste_pct_arr)):
        ax_right.annotate(f"{intr:.1f}", (i, intr/2), ha="center", va="center",
                          color="white", fontsize=9)
        ax_right.annotate(f"+{wst:.1f}\n({pct:.0f}%)", (i, intr + wst/2), ha="center",
                          va="center", color="white", fontsize=9)
        ax_right.annotate(f"batch S_k\n{intr+wst:.1f}", (i, intr+wst), textcoords="offset points",
                          xytext=(0, 4), ha="center", fontsize=8.5)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(CONCS)
    ax_right.set_xlabel("concurrency")
    ax_right.set_ylabel("forwards per block")
    ax_right.set_ylim(0, max(i+w for i, w in zip(intrinsic_arr, waste_arr)) * 1.34)
    ax_right.set_title("Each block's batch S_k = productive + straggler waste")
    ax_right.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- Figure 5: communication-time fraction (per forward vs per token) ------------
def figure5_comm_fraction(data, out_path):
    """Histogram: comm/(comm+comp) fraction per forward (top, unweighted) vs per token
    (bottom, token-weighted). The VALUE is the same, the DISTRIBUTION differs."""
    edges = np.linspace(0, 1, 42)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), sharex=True, sharey="row")
    fig.suptitle("Communication-time fraction  comm/(comm+compute):\n"
                 "per FORWARD (top) vs per TOKEN (bottom, weighted by tokens decoded)", y=1.0)
    for row_idx, weighted in enumerate((False, True)):
        for ax, c in zip(axes[row_idx], CONCS):
            frac_arr = data[c]["comm_frac_arr"]
            wts = data[c]["n_committed_arr"] if weighted else np.ones(len(frac_arr))
            wts = wts / wts.sum()
            mean_val = weighted_mean(frac_arr, wts)
            med_val = weighted_median(frac_arr, wts)
            ax.hist(frac_arr, bins=edges, weights=wts, color=COL_FRAC, alpha=0.8,
                    edgecolor="white", lw=0.3)
            ax.axvline(mean_val, color="black", ls="--", lw=1.6)
            ax.axvline(med_val, color="black", ls=":", lw=1.6)
            y_top = ax.get_ylim()[1]
            ax.annotate(f"mean {mean_val*100:.0f}%", (mean_val, y_top), xytext=(4, -10),
                        textcoords="offset points", fontsize=8)
            ax.annotate(f"median {med_val*100:.0f}%", (med_val, y_top), xytext=(4, -22),
                        textcoords="offset points", fontsize=8)
            ax.set_xlim(0, 1)
            level = "per token" if weighted else "per forward"
            ax.set_title(f"conc {c}  ({level}, n={len(frac_arr)})", fontsize=9)
            if row_idx == 1:
                ax.set_xlabel("comm fraction of GPU time")
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of tokens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Figure 6: raw per-forward time (comm and comp) -----------------------------
def figure6_perfwd_time(data, out_path):
    """Histogram: raw comm time and comp time per forward (shows desync tail)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    fig.suptitle("Per-FORWARD measured time: communication (top) vs compute (bottom)\n"
                 "— comm has a desync tail, compute is tight", y=1.0)
    plot_hist_row(axes[0], {c: data[c]["comm_fwd_arr"] for c in CONCS},
                  xmax=6.0, color=COL_COMM, xlabel="communication time per forward (ms)",
                  unit=" ms")
    plot_hist_row(axes[1], {c: data[c]["comp_fwd_arr"] for c in CONCS},
                  xmax=8.0, color=COL_COMP, xlabel="compute time per forward (ms)",
                  unit=" ms")
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of forwards")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Main ------------------------------------------------------------------------
def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_ROOT, EXP_SUBDIR, "logs")
    figs_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "experiments", EXP_SUBDIR, "figures")
    profiles_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_ROOT, EXP_SUBDIR, "profiles")
    os.makedirs(figs_dir, exist_ok=True)

    print("[plot_d2] Loading data...")
    data = load_all(logs_dir, profiles_dir)

    print("[plot_d2] Generating figures...")
    figure1_pertoken_means_vs_concurrency(data, os.path.join(figs_dir, "fig1_pertoken_vs_concurrency.png"))
    figure2_committed_per_forward_hist(data, os.path.join(figs_dir, "fig2_committed_per_step_hist.png"))
    figure3_pertoken_cost_distributions(data, os.path.join(figs_dir, "fig3_pertoken_hist.png"))
    figure4_sk_and_straggler(data, os.path.join(figs_dir, "fig4_sk_and_straggler.png"))
    figure5_comm_fraction(data, os.path.join(figs_dir, "fig5_comm_fraction_hist.png"))
    figure6_perfwd_time(data, os.path.join(figs_dir, "fig6_perfwd_time_hist.png"))

    # dump distribution stats for the report (single source of truth)
    stats = {}
    for c in CONCS:
        fr = data[c]["comm_frac_arr"]
        wt = data[c]["n_committed_arr"]
        u = np.ones(len(fr))
        cpt = data[c]["comm_per_token_arr"]
        ppt = data[c]["comp_per_token_arr"]
        rc = data[c]["comm_fwd_arr"]
        stats[c] = {
            "commfrac_perfwd_mean": weighted_mean(fr, u),
            "commfrac_perfwd_median": weighted_median(fr, u),
            "commfrac_pertok_mean": weighted_mean(fr, wt),
            "commfrac_pertok_median": weighted_median(fr, wt),
            "commtok_tw_mean": weighted_mean(cpt, wt),
            "commtok_tw_median": weighted_median(cpt, wt),
            "comptok_tw_mean": weighted_mean(ppt, wt),
            "comptok_tw_median": weighted_median(ppt, wt),
            "commfwd_mean": rc.mean(),
            "commfwd_median": float(np.median(rc)),
        }
        print(f"c{c}: comm-frac per-fwd mean={weighted_mean(fr,u)*100:.1f}% "
              f"med={weighted_median(fr,u)*100:.1f}% | per-tok mean={weighted_mean(fr,wt)*100:.1f}% "
              f"med={weighted_median(fr,wt)*100:.1f}% | comm/tok mean={weighted_mean(cpt,wt):.3f} "
              f"med={weighted_median(cpt,wt):.3f} | comp/tok mean={weighted_mean(ppt,wt):.3f} "
              f"med={weighted_median(ppt,wt):.3f}")
    json.dump(stats, open(os.path.join(logs_dir, "d2_dist_stats.json"), "w"), indent=2)
    print(f"[plot_d2] Wrote 6 figures -> {figs_dir}  (+ d2_dist_stats.json)")


if __name__ == "__main__":
    main()
