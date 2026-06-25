#!/usr/bin/env python3
"""D2 figures — per-FORWARD / per-TOKEN distributions, not just means.

For every denoising forward we have (a) its committed-token count (per-step counter
CSV) and (b) its measured comm/comp GPU time (reconstructed per CUDA-graph replay
from the nsys .sqlite). Dividing gives one sample of comm/token and comp/token PER
FORWARD; comm/(comm+comp) gives the communication-time FRACTION per forward. We plot
the full distributions (histograms + mean + median), because a single number or a
"typical..mean" range hides the heavy right tail (rare cross-rank desync stalls).

Per-replay ↔ per-forward join: a forward's realized batch_size is PADDED up to the
nearest captured CUDA-graph size (`cuda_graph_bs`), so we map each graph to a padded
bs by replay-count rank, pool its per-replay (comm,comp), and pair with counter
forwards at that padded bs. comm⊥committed (comm time is shape-fixed + a content-
independent desync tail — validated by the per-replay CV in parse_d2.py), so pairing
by padded-bs is statistically valid. The comm-FRACTION needs no committed at all
(it is comm/(comm+comp) of each replay), so it is exact per forward.

  fig1  per-token comm/comp MEANS vs concurrency (summary line)
  fig2  tokens-committed-per-forward distribution (the per-token denominator)
  fig3  comm/token and comp/token per-FORWARD distributions (hist + mean + median)
  fig4  intrinsic s_k distribution (left) + batch S_k = intrinsic + straggler (right)
  fig5  communication-time FRACTION per forward (= per token) — hist + mean + median
  fig6  raw comm-time and comp-time per forward (hist; shows the comm desync tail)

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

CONCS = [4, 8, 16]
COL_COMM, COL_COMP, COL_TOK = "#d62728", "#1f77b4", "#2ca02c"
COL_WASTE, COL_GOOD, COL_FRAC = "#d62728", "#2ca02c", "#9467bd"
REPO = os.environ.get("REPO", "/root/sglang_a100/sglang")
DATA_ROOT = os.environ.get("DATA_ROOT", "/cephfs/shared/wxli/sglang-dllm")
EXP = "profiling/dllm/d2_sk_amplification"
# captured CUDA-graph batch sizes (server_args cuda_graph_bs) — forwards pad up to these
CAPS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
COMM_RE = re.compile(
    r"nccl|all.?reduce|all.?gather|all.?to.?all|reduce.?scatter|sendrecv|nvshmem", re.I)


def pad_bs(b):
    for g in CAPS:
        if g >= b:
            return g
    return CAPS[-1]


def tag(c):
    return f"d2_h100_tp4_c{c}"


def per_replay_by_graph(db):
    """Per graphId: list of (comm_ms, comp_ms) per replay (rep rank). Each node fires
    once/replay; sort a node's instances by start → index = replay."""
    cur = sqlite3.connect(db).cursor()
    dev = cur.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphId,k.graphNodeId,k.start,k.end-k.start,s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphId IS NOT NULL", (dev,)).fetchall()
    bg = defaultdict(lambda: defaultdict(list))
    for gid, node, start, dur, nm in rows:
        bg[gid][node].append((start, dur, nm))
    out = {}
    for gid, nodes in bg.items():
        n = max(len(v) for v in nodes.values())
        comm, comp = [0.0] * n, [0.0] * n
        for inst in nodes.values():
            inst.sort()
            for i, (s, d, nm) in enumerate(inst):
                if i >= n:
                    break
                if COMM_RE.search(nm):
                    comm[i] += d
                else:
                    comp[i] += d
        out[gid] = [(comm[i] / 1e6, comp[i] / 1e6) for i in range(n)]
    return out


def load(logs, profiles):
    """Return per-conc: metrics, committed[], intrinsic_sk[], and per-FORWARD samples
    of comm/token, comp/token, comm-fraction, raw comm, raw comp."""
    D = {}
    for c in CONCS:
        base = os.path.join(logs, tag(c) + "_blocks")
        m = json.load(open(base + "_d2metrics.json"))
        ps = [(int(r["batch_size"]), int(r["committed"]))
              for r in csv.DictReader(open(base + "_perstep.csv"))]
        sk = [int(r["finish_step"]) + 1 if int(r["finish_step"]) >= 0 else int(r["S_k"])
              for r in csv.DictReader(open(base + ".csv"))]
        graphs = per_replay_by_graph(os.path.join(profiles, tag(c) + ".sqlite"))
        # map graphId -> padded bs by replay-count rank vs counter padded-bs counts
        padcount = Counter(pad_bs(b) for b, _ in ps)
        gids = sorted(graphs, key=lambda g: -len(graphs[g]))
        pbs = [b for b, _ in padcount.most_common()]
        gid2bs = {g: b for g, b in zip(gids, pbs)}
        pools = defaultdict(list)
        for g, b in gid2bs.items():
            pools[b].extend(graphs[g])
        avail = sorted(pools)
        # one sample per FORWARD: (comm_time, comp_time, n_tokens) → per-token ratios
        # (your definition). w = n_tokens is the weight that turns a per-FORWARD
        # distribution into a per-TOKEN one (a forward decoding n tokens counts n×).
        cpt, ppt, frac, rc, rp, w = [], [], [], [], [], []
        cyc = defaultdict(int)
        for b, com in ps:
            if com <= 0:
                continue
            p = pad_bs(b)
            if p not in pools:                       # nearest available pool
                p = min(avail, key=lambda x: abs(x - p))
            pool = pools[p]
            comm_i, comp_i = pool[cyc[p] % len(pool)]
            cyc[p] += 1
            cpt.append(comm_i / com)                 # comm per token  = comm_time/n
            ppt.append(comp_i / com)                 # comp per token  = comp_time/n
            frac.append(comm_i / (comm_i + comp_i) if comm_i + comp_i else 0.0)
            rc.append(comm_i)
            rp.append(comp_i)
            w.append(com)                            # token weight n
        D[c] = dict(m=m, committed=[com for _, com in ps if com > 0], sk=sk,
                    cpt=cpt, ppt=ppt, frac=frac, w=w, rc=rc, rp=rp,
                    dom_bs=gids and gid2bs[gids[0]])
    return D


# ----------------------------------------------------------------------------- utils
def wmean(xs, w):
    return float(np.average(xs, weights=w)) if len(xs) else 0.0


def wmedian(xs, w):
    xs = np.asarray(xs, float)
    w = np.asarray(w, float)
    o = np.argsort(xs)
    xs, cw = xs[o], np.cumsum(w[o])
    return float(xs[np.searchsorted(cw, 0.5 * cw[-1])])


def stat_lines(ax, mean, med, unit=""):
    ax.axvline(mean, color="black", ls="--", lw=1.6)
    ax.axvline(med, color="black", ls=":", lw=1.6)
    y = ax.get_ylim()[1]
    ax.annotate(f"mean {mean:.3g}{unit}", (mean, y), xytext=(4, -10),
                textcoords="offset points", fontsize=8)
    ax.annotate(f"median {med:.3g}{unit}", (med, y), xytext=(4, -22),
                textcoords="offset points", fontsize=8)


def hist_row(axes, data_by_c, xmax, color, xlabel, unit="", bins=44, weights_by_c=None):
    """Histogram per concurrency. If weights_by_c given, each sample is weighted (e.g.
    by committed tokens → a distribution over DELIVERED TOKENS whose mean is the
    token-weighted cost); else unweighted (a distribution over FORWARDS)."""
    edges = np.linspace(0, xmax, bins)
    for ax, c in zip(axes, CONCS):
        d = np.array(data_by_c[c], float)
        wt = np.array(weights_by_c[c], float) if weights_by_c else np.ones(len(d))
        wt = wt / wt.sum()
        over = float(wt[d > xmax].sum())
        ax.hist(np.clip(d, 0, xmax), bins=edges, weights=wt,
                color=color, alpha=0.8, edgecolor="white", lw=0.3)
        stat_lines(ax, wmean(d, wt), wmedian(d, wt), unit=unit)
        t = f"conc {c}  (n={len(d)})"
        if over > 0.005:
            t += f"\n{100*over:.0f}% > {xmax:g} (tail to {d.max():.2g})"
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------- figures
def fig1(D, out):
    x = np.arange(len(CONCS))
    # token-weighted means from the per-replay distribution (same source as fig3/L4)
    comm = [wmean(D[c]["cpt"], D[c]["w"]) for c in CONCS]
    comp = [wmean(D[c]["ppt"], D[c]["w"]) for c in CONCS]
    tok = [st.mean(D[c]["committed"]) for c in CONCS]
    fig, axL = plt.subplots(figsize=(6.4, 4.2))
    axL.plot(x, comp, "o-", color=COL_COMP, lw=2, ms=7, label="compute / token")
    axL.plot(x, comm, "o-", color=COL_COMM, lw=2, ms=7, label="exposed comm / token")
    for xi, (a, b) in enumerate(zip(comm, comp)):
        axL.annotate(f"{b:.2f}", (xi, b), textcoords="offset points", xytext=(0, 8),
                     ha="center", color=COL_COMP, fontsize=9)
        axL.annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(0, -14),
                     ha="center", color=COL_COMM, fontsize=9)
    axL.set_ylabel("ms per output token (token-weighted mean)")
    axL.set_xlabel("concurrency (= max-running-requests = dominant batch size)")
    axL.set_ylim(0, max(comp) * 1.25)
    axR = axL.twinx()
    axR.plot(x, tok, "s--", color=COL_TOK, lw=2, ms=7, label="tokens / forward")
    for xi, t in enumerate(tok):
        axR.annotate(f"{t:.1f}", (xi, t), textcoords="offset points", xytext=(0, 9),
                     ha="center", color=COL_TOK, fontsize=9)
    axR.set_ylabel("tokens committed per forward", color=COL_TOK)
    axR.tick_params(axis="y", labelcolor=COL_TOK)
    axR.set_ylim(0, max(tok) * 1.32)
    axL.set_xticks(x)
    axL.set_xticklabels(CONCS)
    l1, lb1 = axL.get_legend_handles_labels()
    l2, lb2 = axR.get_legend_handles_labels()
    axL.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=9, framealpha=0.9)
    axL.set_title("Per-token comm/compute (means) fall with concurrency")
    axL.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig2(D, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True, sharey=True)
    fig.suptitle("Tokens committed per forward (the per-token denominator)", y=1.02)
    hist_row(axes, {c: D[c]["committed"] for c in CONCS}, 50, COL_TOK,
             "tokens committed in one forward")
    axes[0].set_ylabel("share of forwards")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig3(D, out):
    """Per-token cost distribution, token-weighted (each forward weighted by the
    tokens it delivered → distribution over DELIVERED TOKENS; its mean equals the
    token-weighted headline cost). comm = measured comm-time/forward ÷ committed."""
    w = {c: D[c]["w"] for c in CONCS}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    fig.suptitle("Distribution over delivered tokens of cost per output token "
                 "(measured comm/comp time per forward ÷ tokens decoded; token-weighted)", y=1.0)
    hist_row(axes[0], {c: D[c]["cpt"] for c in CONCS}, 2.0, COL_COMM,
             "exposed comm per output token (ms)", weights_by_c=w)
    hist_row(axes[1], {c: D[c]["ppt"] for c in CONCS}, 4.0, COL_COMP,
             "compute per output token (ms)", weights_by_c=w)
    axes[0][0].set_ylabel("share of delivered tokens")
    axes[1][0].set_ylabel("share of delivered tokens")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig4(D, out):
    m = {c: D[c]["m"] for c in CONCS}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0))
    pooled = np.concatenate([np.array(D[c]["sk"]) for c in CONCS])
    axL.hist(pooled, bins=np.arange(0, 34, 1),
             weights=np.ones(len(pooled)) / len(pooled),
             color="#7f7f7f", alpha=0.75, edgecolor="white", lw=0.4)
    for c, col in zip(CONCS, ["#9467bd", "#ff7f0e", "#17becf"]):
        mu = m[c]["intrinsic_sk_mean"]
        axL.axvline(mu, color=col, ls="--", lw=1.8, label=f"conc {c}: mean {mu:.1f}")
    axL.set_xlabel("intrinsic s_k (block's own denoising steps = finish_step+1)")
    axL.set_ylabel("share of blocks")
    axL.set_title("Intrinsic s_k is content-driven (pooled; means ~coincide)")
    axL.legend(fontsize=9)
    axL.grid(True, axis="y", alpha=0.3)
    x = np.arange(len(CONCS))
    good = [m[c]["intrinsic_sk_mean"] for c in CONCS]
    waste = [m[c]["straggler_waste_fwd"] for c in CONCS]
    pct = [m[c]["straggler_waste_pct"] for c in CONCS]
    axR.bar(x, good, color=COL_GOOD, label="productive (intrinsic s_k)")
    axR.bar(x, waste, bottom=good, color=COL_WASTE,
            label="straggler waste (forwards after finish)")
    for xi, (g, w, p) in enumerate(zip(good, waste, pct)):
        axR.annotate(f"{g:.1f}", (xi, g / 2), ha="center", va="center", color="white", fontsize=9)
        axR.annotate(f"+{w:.1f}\n({p:.0f}%)", (xi, g + w / 2), ha="center", va="center",
                     color="white", fontsize=9)
        axR.annotate(f"batch S_k\n{g + w:.1f}", (xi, g + w), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5)
    axR.set_xticks(x)
    axR.set_xticklabels(CONCS)
    axR.set_xlabel("concurrency")
    axR.set_ylabel("forwards per block")
    axR.set_ylim(0, max(g + w for g, w in zip(good, waste)) * 1.34)
    axR.set_title("Each block's batch S_k = productive + straggler waste")
    axR.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig5(D, out):
    """Communication-time fraction = comm/(comm+comp). The fraction VALUE is the same
    whether you form it per forward or from the per-token ratios (the n_tokens cancels:
    (comm/n)/((comm/n)+(comp/n)) = comm/(comm+comp)) — but the DISTRIBUTION is not the
    same: per FORWARD each forward is one sample (top); per TOKEN each forward is
    weighted by the n tokens it decoded (bottom), since a forward decoding 40 tokens
    represents 40 tokens of the workload. We plot both so the difference is explicit."""
    edges = np.linspace(0, 1, 42)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), sharex=True, sharey="row")
    fig.suptitle("Communication-time fraction  comm/(comm+compute):  "
                 "per FORWARD (top) vs per TOKEN (bottom, weighted by tokens decoded)", y=1.0)
    for row, weighted in enumerate((False, True)):
        for ax, c in zip(axes[row], CONCS):
            d = np.array(D[c]["frac"], float)
            wt = np.array(D[c]["w"], float) if weighted else np.ones(len(d))
            wt = wt / wt.sum()
            mean, med = wmean(d, wt), wmedian(d, wt)
            ax.hist(d, bins=edges, weights=wt, color=COL_FRAC, alpha=0.8,
                    edgecolor="white", lw=0.3)
            ax.axvline(mean, color="black", ls="--", lw=1.6)
            ax.axvline(med, color="black", ls=":", lw=1.6)
            y = ax.get_ylim()[1]
            ax.annotate(f"mean {mean*100:.0f}%", (mean, y), xytext=(4, -10),
                        textcoords="offset points", fontsize=8)
            ax.annotate(f"median {med*100:.0f}%", (med, y), xytext=(4, -22),
                        textcoords="offset points", fontsize=8)
            ax.set_xlim(0, 1)
            lvl = "per token" if weighted else "per forward"
            ax.set_title(f"conc {c}  ({lvl}, n={len(d)})", fontsize=9)
            if row == 1:
                ax.set_xlabel("comm fraction of GPU time")
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of tokens")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig6(D, out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    fig.suptitle("Per-FORWARD measured time: communication (top) vs compute (bottom) — "
                 "comm has a desync tail, compute is tight", y=1.0)
    hist_row(axes[0], {c: D[c]["rc"] for c in CONCS}, 6.0, COL_COMM,
             "communication time per forward (ms)", unit=" ms")
    hist_row(axes[1], {c: D[c]["rp"] for c in CONCS}, 8.0, COL_COMP,
             "compute time per forward (ms)", unit=" ms")
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of forwards")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    logs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_ROOT, EXP, "logs")
    figs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "experiments", EXP, "figures")
    profiles = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_ROOT, EXP, "profiles")
    os.makedirs(figs, exist_ok=True)
    D = load(logs, profiles)
    fig1(D, os.path.join(figs, "fig1_pertoken_vs_concurrency.png"))
    fig2(D, os.path.join(figs, "fig2_committed_per_step_hist.png"))
    fig3(D, os.path.join(figs, "fig3_pertoken_hist.png"))
    fig4(D, os.path.join(figs, "fig4_sk_and_straggler.png"))
    fig5(D, os.path.join(figs, "fig5_comm_fraction_hist.png"))
    fig6(D, os.path.join(figs, "fig6_perfwd_time_hist.png"))
    # numeric recap + dump distribution stats (single source of truth for the report)
    stats = {}
    for c in CONCS:
        fr, w = D[c]["frac"], D[c]["w"]
        u = [1.0] * len(fr)
        cpt, ppt, rc = D[c]["cpt"], D[c]["ppt"], D[c]["rc"]
        stats[c] = dict(
            commfrac_perfwd_mean=wmean(fr, u), commfrac_perfwd_median=wmedian(fr, u),
            commfrac_pertok_mean=wmean(fr, w), commfrac_pertok_median=wmedian(fr, w),
            commtok_twmean=wmean(cpt, w), commtok_twmedian=wmedian(cpt, w),
            comptok_twmean=wmean(ppt, w), comptok_twmedian=wmedian(ppt, w),
            commfwd_mean=st.mean(rc), commfwd_median=st.median(rc))
        print(f"c{c}: comm-frac per-fwd mean={wmean(fr,u)*100:.1f}% med={wmedian(fr,u)*100:.1f}% | "
              f"per-tok mean={wmean(fr,w)*100:.1f}% med={wmedian(fr,w)*100:.1f}% | "
              f"comm/tok mean={wmean(cpt,w):.3f} med={wmedian(cpt,w):.3f} | "
              f"comp/tok mean={wmean(ppt,w):.3f} med={wmedian(ppt,w):.3f}")
    json.dump(stats, open(os.path.join(logs, "d2_dist_stats.json"), "w"), indent=2)
    print(f"[plot_d2] wrote 6 figures -> {figs}  (+ d2_dist_stats.json)")


if __name__ == "__main__":
    main()
