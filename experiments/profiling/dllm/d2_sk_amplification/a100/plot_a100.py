#!/usr/bin/env python3
"""D2 / A100 figures — the SAME analysis as the H100 D2 leaf (plot_d2.py), but the
communication time is PROJECTED from analytic volume onto A100 NVLink, while the
compute time is the REAL A100 measurement. So every comm/comp number answers: what
would the dLLM's comm/compute balance be if this A100 box had NVLink?

  comm/forward(bs)  = (all-reduce bus + vocab all-gather bus)(bs) / A100-NVLink busbw   [PROJECTED, deterministic per bs]
  compute/forward   = measured A100 per-CUDA-graph-replay compute time                  [REAL]
  per token         = the above / tokens committed that forward (§3 counter)            [token-weighted]
  comm fraction     = comm / (comm + compute)

No-GPU: reads SAVED per-bs *_a100metrics.json + the per-step counter CSVs + the nsys
.sqlite (compute per replay only), never re-runs the model. Mirrors D2's six figures.

  fig1  per-token comm/comp MEANS vs batch size (+ tokens/forward)        [cf. D2 fig1]
  fig2  tokens-committed-per-forward distribution                          [cf. D2 fig2]
  fig3  comm/token & comp/token per-token distributions (hist+mean+median) [cf. D2 fig3]
  fig4  intrinsic s_k distribution + batch S_k = intrinsic + straggler     [cf. D2 fig4]
  fig5  communication-time FRACTION per forward vs per token               [cf. D2 fig5]
  fig6  per-forward comm (projected) & compute (measured) time distribution[cf. D2 fig6]

Usage: plot_a100.py [logs_dir] [figures_dir] [profiles_dir]
"""
import csv
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_a100 import (BUSBW_REFS, BUSBW_HEADLINE, volume_fwd, ag_volume_fwd,
                        _graph_nodes, _bucket_graphs, COLLECTIVE_RE, ALLGATHER_RE)

BSES = [1, 4, 8, 16]
COL_COMM, COL_COMP, COL_TOK = "#d62728", "#1f77b4", "#2ca02c"
COL_WASTE, COL_GOOD, COL_FRAC = "#d62728", "#2ca02c", "#9467bd"
REPO = os.environ.get("REPO", "/root/sglang_a100/sglang")
DATA_ROOT = os.environ.get("DATA_ROOT", "/cephfs/shared/wxli/sglang-dllm")
EXP = "profiling/dllm/d2_sk_amplification/a100"
CAPS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
BUSBW = BUSBW_REFS[BUSBW_HEADLINE]


def pad_bs(b):
    for g in CAPS:
        if g >= b:
            return g
    return CAPS[-1]


def tag(c):
    return f"d2_a100_tp4_c{c}"


def proj_comm_ms(bs):
    """PROJECTED A100-NVLink comm time/forward (ms) = total bus traffic / busbw."""
    return (volume_fwd(bs)[1] + ag_volume_fwd(bs)[1]) / BUSBW * 1e3


def comp_per_replay_by_bs(db):
    """Per graph (replay-count bucket) -> list of measured compute ms per replay, for
    the rep rank. Same dominant-graph reconstruction the parser uses (graphNodeId)."""
    bynode, dev = _graph_nodes(db)
    out = {}
    for R, nodes in _bucket_graphs(bynode).items():
        comp = [0.0] * R
        for node in nodes:
            for i, (s, d, nm) in enumerate(sorted(bynode[node])):
                if i >= R:
                    break
                if not COLLECTIVE_RE.search(nm):
                    comp[i] += d
        out[R] = [x / 1e6 for x in comp]
    return out


def load(logs, profiles):
    D = {}
    for c in BSES:
        base = os.path.join(logs, tag(c) + "_blocks")
        mp = base + "_a100metrics.json"
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        ps = [(int(r["batch_size"]), int(r["committed"]))
              for r in csv.DictReader(open(base + "_perstep.csv"))]
        sk = [int(r["finish_step"]) + 1 if int(r["finish_step"]) >= 0 else int(r["S_k"])
              for r in csv.DictReader(open(base + ".csv"))]
        comps = comp_per_replay_by_bs(os.path.join(profiles, tag(c) + ".sqlite"))
        # map graph (by replay-count rank) to a padded bs (by frequency rank), like D2
        padcount = Counter(pad_bs(b) for b, _ in ps)
        Rs = sorted(comps, key=lambda R: -R * len(comps[R]))   # heaviest graph first
        pbs = [b for b, _ in padcount.most_common()]
        R2bs = {R: b for R, b in zip(Rs, pbs)}
        pools = defaultdict(list)
        for R, b in R2bs.items():
            pools[b].extend(comps[R])
        avail = sorted(pools)
        # one sample per FORWARD: comm projected (deterministic by bs), comp measured
        cpt, ppt, frac, rc, rp, w = [], [], [], [], [], []
        cyc = defaultdict(int)
        for b, com in ps:
            if com <= 0:
                continue
            p = pad_bs(b)
            pool = pools[p] if p in pools else pools[min(avail, key=lambda x: abs(x - p))]
            comp_i = pool[cyc[p] % len(pool)]
            cyc[p] += 1
            comm_i = proj_comm_ms(p)                  # projected comm at this forward's bs
            cpt.append(comm_i / com)
            ppt.append(comp_i / com)
            frac.append(comm_i / (comm_i + comp_i) if comm_i + comp_i else 0.0)
            rc.append(comm_i)
            rp.append(comp_i)
            w.append(com)
        D[c] = dict(m=m, committed=[com for _, com in ps if com > 0], sk=sk,
                    cpt=cpt, ppt=ppt, frac=frac, w=w, rc=rc, rp=rp)
    return D


def wmean(xs, w):
    return float(np.average(xs, weights=w)) if len(xs) else 0.0


def wmedian(xs, w):
    xs, w = np.asarray(xs, float), np.asarray(w, float)
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


def hist_row(axes, data_by_c, xmax, color, xlabel, unit="", bins=44, weights_by_c=None, cs=BSES):
    edges = np.linspace(0, xmax, bins)
    for ax, c in zip(axes, cs):
        d = np.array(data_by_c[c], float)
        wt = np.array(weights_by_c[c], float) if weights_by_c else np.ones(len(d))
        wt = wt / wt.sum()
        over = float(wt[d > xmax].sum())
        ax.hist(np.clip(d, 0, xmax), bins=edges, weights=wt, color=color, alpha=0.8,
                edgecolor="white", lw=0.3)
        stat_lines(ax, wmean(d, wt), wmedian(d, wt), unit=unit)
        t = f"bs {c}  (n={len(d)})"
        if over > 0.005:
            t += f"\n{100*over:.0f}% > {xmax:g} (tail to {d.max():.2g})"
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(xlabel)


def fig1(D, out, cs):
    x = np.arange(len(cs))
    comm = [wmean(D[c]["cpt"], D[c]["w"]) for c in cs]
    comp = [wmean(D[c]["ppt"], D[c]["w"]) for c in cs]
    tok = [st.mean(D[c]["committed"]) for c in cs]
    fig, axL = plt.subplots(figsize=(6.8, 4.3))
    axL.plot(x, comp, "o-", color=COL_COMP, lw=2, ms=7, label="compute / token (measured A100)")
    axL.plot(x, comm, "o-", color=COL_COMM, lw=2, ms=7, label="comm / token (projected NVLink)")
    for xi, (a, b) in enumerate(zip(comm, comp)):
        axL.annotate(f"{b:.2f}", (xi, b), textcoords="offset points", xytext=(0, 8),
                     ha="center", color=COL_COMP, fontsize=9)
        axL.annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(0, -14),
                     ha="center", color=COL_COMM, fontsize=9)
    axL.set_ylabel("ms per output token (token-weighted mean)")
    axL.set_xlabel("batch size (= max-running-requests = dominant bs)")
    axL.set_ylim(0, max(comp) * 1.25)
    axR = axL.twinx()
    axR.plot(x, tok, "s--", color=COL_TOK, lw=2, ms=7, label="tokens / forward")
    for xi, t in enumerate(tok):
        axR.annotate(f"{t:.1f}", (xi, t), textcoords="offset points", xytext=(0, 9),
                     ha="center", color=COL_TOK, fontsize=9)
    axR.set_ylabel("tokens committed per forward", color=COL_TOK)
    axR.tick_params(axis="y", labelcolor=COL_TOK)
    axR.set_ylim(0, max(tok) * 1.32)
    axL.set_xticks(x); axL.set_xticklabels(cs)
    l1, lb1 = axL.get_legend_handles_labels()
    l2, lb2 = axR.get_legend_handles_labels()
    axL.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=9, framealpha=0.9)
    axL.set_title("Per-token compute (measured A100) & comm (projected NVLink) vs batch size")
    axL.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig2(D, out, cs):
    fig, axes = plt.subplots(1, len(cs), figsize=(2.9 * len(cs), 3.2), sharex=True, sharey=True)
    fig.suptitle("Tokens committed per forward (the per-token denominator)", y=1.02)
    hist_row(axes, {c: D[c]["committed"] for c in cs}, 50, COL_TOK,
             "tokens committed in one forward", cs=cs)
    axes[0].set_ylabel("share of forwards")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig3(D, out, cs):
    w = {c: D[c]["w"] for c in cs}
    fig, axes = plt.subplots(2, len(cs), figsize=(2.9 * len(cs), 6.2), sharey="row")
    fig.suptitle("Distribution over delivered tokens: comm/token (projected NVLink, top) & "
                 "compute/token (measured A100, bottom); token-weighted", y=1.0)
    hist_row(axes[0], {c: D[c]["cpt"] for c in cs}, 0.25, COL_COMM,
             "projected comm per output token (ms)", weights_by_c=w, cs=cs)
    hist_row(axes[1], {c: D[c]["ppt"] for c in cs}, 4.0, COL_COMP,
             "measured compute per output token (ms)", weights_by_c=w, cs=cs)
    axes[0][0].set_ylabel("share of delivered tokens")
    axes[1][0].set_ylabel("share of delivered tokens")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig4(D, out, cs):
    m = {c: D[c]["m"] for c in cs}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0))
    pooled = np.concatenate([np.array(D[c]["sk"]) for c in cs])
    axL.hist(pooled, bins=np.arange(0, 34, 1), weights=np.ones(len(pooled)) / len(pooled),
             color="#7f7f7f", alpha=0.75, edgecolor="white", lw=0.4)
    for c, col in zip(cs, ["#9467bd", "#ff7f0e", "#17becf", "#8c564b"]):
        mu = m[c]["intrinsic_sk_mean"]
        axL.axvline(mu, color=col, ls="--", lw=1.8, label=f"bs {c}: mean {mu:.1f}")
    axL.set_xlabel("intrinsic s_k (block's own denoising steps = finish_step+1)")
    axL.set_ylabel("share of blocks")
    axL.set_title("Intrinsic s_k is content-driven (pooled; means ~coincide)")
    axL.legend(fontsize=9); axL.grid(True, axis="y", alpha=0.3)
    x = np.arange(len(cs))
    good = [m[c]["intrinsic_sk_mean"] for c in cs]
    batch = [m[c]["batch_sk_mean"] for c in cs]
    waste = [b - g for g, b in zip(good, batch)]
    pct = [m[c]["straggler_waste_pct"] for c in cs]
    axR.bar(x, good, color=COL_GOOD, label="productive (intrinsic s_k)")
    axR.bar(x, waste, bottom=good, color=COL_WASTE, label="straggler waste (forwards after finish)")
    for xi, (g, wv, p) in enumerate(zip(good, waste, pct)):
        axR.annotate(f"{g:.1f}", (xi, g / 2), ha="center", va="center", color="white", fontsize=9)
        if wv > 0.3:
            axR.annotate(f"+{wv:.1f}\n({p:.0f}%)", (xi, g + wv / 2), ha="center", va="center",
                         color="white", fontsize=9)
        axR.annotate(f"batch S_k\n{g + wv:.1f}", (xi, g + wv), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5)
    axR.set_xticks(x); axR.set_xticklabels(cs); axR.set_xlabel("batch size")
    axR.set_ylabel("forwards per block")
    axR.set_ylim(0, max(batch) * 1.34)
    axR.set_title("Each block's batch S_k = productive + straggler waste")
    axR.legend(fontsize=9, loc="upper left", framealpha=0.95)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig5(D, out, cs):
    edges = np.linspace(0, 0.3, 42)
    fig, axes = plt.subplots(2, len(cs), figsize=(2.9 * len(cs), 6.0), sharex=True, sharey="row")
    fig.suptitle("Projected communication-time fraction  comm/(comm+compute)  (comm projected "
                 "NVLink, compute measured A100):  per FORWARD (top) vs per TOKEN (bottom)", y=1.0)
    for row, weighted in enumerate((False, True)):
        for ax, c in zip(axes[row], cs):
            d = np.array(D[c]["frac"], float)
            wt = np.array(D[c]["w"], float) if weighted else np.ones(len(d))
            wt = wt / wt.sum()
            mean, med = wmean(d, wt), wmedian(d, wt)
            ax.hist(d, bins=edges, weights=wt, color=COL_FRAC, alpha=0.8, edgecolor="white", lw=0.3)
            ax.axvline(mean, color="black", ls="--", lw=1.6)
            ax.axvline(med, color="black", ls=":", lw=1.6)
            y = ax.get_ylim()[1]
            ax.annotate(f"mean {mean*100:.1f}%", (mean, y), xytext=(4, -10),
                        textcoords="offset points", fontsize=8)
            ax.annotate(f"median {med*100:.1f}%", (med, y), xytext=(4, -22),
                        textcoords="offset points", fontsize=8)
            ax.set_xlim(0, 0.3)
            ax.set_title(f"bs {c}  ({'per token' if weighted else 'per forward'}, n={len(d)})", fontsize=9)
            if row == 1:
                ax.set_xlabel("projected comm fraction of forward")
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of tokens")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig6(D, out, cs):
    fig, axes = plt.subplots(2, len(cs), figsize=(2.9 * len(cs), 6.2), sharey="row")
    fig.suptitle("Per-forward time: projected NVLink comm (top, deterministic by bs) vs "
                 "measured A100 compute (bottom, a near-spike)", y=1.0)
    hist_row(axes[0], {c: D[c]["rc"] for c in cs}, 1.2, COL_COMM,
             "projected comm time per forward (ms)", unit=" ms", cs=cs)
    hist_row(axes[1], {c: D[c]["rp"] for c in cs}, 16.0, COL_COMP,
             "measured compute time per forward (ms)", unit=" ms", cs=cs)
    axes[0][0].set_ylabel("share of forwards")
    axes[1][0].set_ylabel("share of forwards")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    logs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_ROOT, EXP, "logs")
    figs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "experiments", EXP, "figures")
    profiles = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_ROOT, EXP, "profiles")
    os.makedirs(figs, exist_ok=True)
    D = load(logs, profiles)
    if not D:
        sys.exit("no *_a100metrics.json found — run run_a100.sh first")
    cs = [c for c in BSES if c in D]
    fig1(D, os.path.join(figs, "fig1_pertoken_vs_bs.png"), cs)
    fig2(D, os.path.join(figs, "fig2_committed_per_step_hist.png"), cs)
    fig3(D, os.path.join(figs, "fig3_pertoken_hist.png"), cs)
    fig4(D, os.path.join(figs, "fig4_sk_and_straggler.png"), cs)
    fig5(D, os.path.join(figs, "fig5_comm_fraction_hist.png"), cs)
    fig6(D, os.path.join(figs, "fig6_perfwd_time_hist.png"), cs)
    stats = {}
    for c in cs:
        fr, w = D[c]["frac"], D[c]["w"]
        u = [1.0] * len(fr)
        stats[c] = dict(
            commfrac_perfwd_mean=wmean(fr, u), commfrac_perfwd_median=wmedian(fr, u),
            commfrac_pertok_mean=wmean(fr, w), commfrac_pertok_median=wmedian(fr, w),
            commtok_twmean=wmean(D[c]["cpt"], w), comptok_twmean=wmean(D[c]["ppt"], w),
            proj_comm_fwd_ms=D[c]["m"]["proj_comm_fwd_ms"], comp_fwd_ms=D[c]["m"]["comp_fwd_ms"],
            comm_frac_pct=D[c]["m"]["comm_frac_pct"], tot_bus_MB=D[c]["m"]["tot_bus_bytes"] / 1e6,
            committed_per_fwd=D[c]["m"]["committed_per_step_mean"],
            intrinsic_sk=D[c]["m"]["intrinsic_sk_mean"], straggler_waste_pct=D[c]["m"]["straggler_waste_pct"])
        print(f"bs{c}: projComm={stats[c]['proj_comm_fwd_ms']:.3f}ms measComp={stats[c]['comp_fwd_ms']:.3f}ms "
              f"frac per-fwd mean={wmean(fr,u)*100:.1f}% med={wmedian(fr,u)*100:.1f}% | "
              f"per-tok mean={wmean(fr,w)*100:.1f}% | comm/tok={wmean(D[c]['cpt'],w):.4f} comp/tok={wmean(D[c]['ppt'],w):.3f}")
    json.dump(stats, open(os.path.join(logs, "a100_dist_stats.json"), "w"), indent=2)
    print(f"[plot_a100] wrote 6 figures -> {figs}  (+ a100_dist_stats.json)")


if __name__ == "__main__":
    main()
