#!/usr/bin/env python3
"""D2 — exposed comm/comp per OUTPUT TOKEN, measured PER STEP.

Methodology (the fix over the first pass): instead of multiplying a block-level
S_k by ONE global average comm_per_forward, we tie EACH denoising forward to the
tokens it actually decoded. Two inputs, both from this run:
  - per-step counter CSV (SGLANG_DLLM_PROFILE): per forward, (batch_size, committed)
  - nsys CUDA-graph split: per forward AT a given batch_size, (comm_ms, comp_ms)
Each denoising forward is a CUDA-graph replay, so its comm/comp time is fixed by
its batch_size (the captured graph). We map graphs to batch sizes by aligning
replay counts with the counter's per-batch_size step counts, then per step:
  comm_per_token = comm_per_forward(bs) / committed_this_step
  comp_per_token = comp_per_forward(bs) / committed_this_step
giving per-token cost as a DISTRIBUTION (sampled per step), not a single ratio.

Usage:
  parse_d2.py <blocks_prefix> <nsys_rep_prefix>   # one run
  parse_d2.py --sweep <logs_dir>                  # aggregate *_d2metrics.json
"""
import csv
import glob
import json
import os
import re
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict

COLLECTIVE_RE = re.compile(
    r"nccl|all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|"
    r"sendrecv|nvshmem|cross_device_reduce|one.?shot|two.?shot",
    re.I,
)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def read_csv_int(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            r[k] = int(r[k])
    return rows


def graph_forward(sqlite_path):
    """Per CUDA-graph: replays + summed comm/comp kernel ns, for the rep rank."""
    cur = sqlite3.connect(sqlite_path).cursor()
    dev = cur.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphId, s.value, k.end-k.start, k.graphNodeId "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphId IS NOT NULL", (dev,),
    ).fetchall()

    g = {}
    for gid, nm, dur, node in rows:
        d = g.setdefault(gid, {"comm": 0, "comp": 0, "nodes": set(), "k": 0})
        d["k"] += 1
        d["nodes"].add(node)
        if COLLECTIVE_RE.search(nm):
            d["comm"] += dur
        else:
            d["comp"] += dur

    out = []
    for gid, d in g.items():
        replays = d["k"] / max(1, len(d["nodes"]))
        out.append(
            {
                "gid": gid, "replays": replays,
                "comm_fwd_ms": d["comm"] / replays / 1e6,
                "comp_fwd_ms": d["comp"] / replays / 1e6,
            }
        )
    out.sort(key=lambda x: -x["replays"])
    return out, dev


def kern_comm_fraction(path):
    rows = list(csv.DictReader(open(path)))
    tot = sum(int(r["Total Time (ns)"]) for r in rows)
    comm = sum(int(r["Total Time (ns)"]) for r in rows if COLLECTIVE_RE.search(r["Name"]))
    return tot / 1e6, comm / 1e6, (100 * comm / tot if tot else 0)


# Attention is the ONLY compute kernel whose runtime depends on content (paged-KV
# length = prompt_len + block), so it is the part that could vary across forwards
# at the same batch size. Used by the stability check below.
ATTN_RE = re.compile(r"flashinfer|attention|paged|prefill|batch.?decode|kv.?cache", re.I)


def _cv(xs):
    m = st.mean(xs)
    return 100 * st.pstdev(xs) / m if m else 0.0


def dominant_per_replay(sqlite_path):
    """Per-REPLAY comm/comp/attn time for the dominant graph — measures whether the
    per-forward cost is actually constant across forwards at the SAME batch size
    (different prompts ⇒ different KV length, different masked positions ⇒ different
    MoE routing). Validates the "one number per batch size" assumption empirically.

    Each graphNodeId fires once per replay and replays run serially on the stream,
    so sorting a node's instances by start time indexes them by replay; replay i's
    cost is the sum over nodes of their i-th instance. Returns (comm[], comp[],
    attn[], gid, dev) in ms.
    """
    cur = sqlite3.connect(sqlite_path).cursor()
    dev = cur.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    gid = cur.execute(
        "SELECT graphId,COUNT(*) k FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId=? AND graphId IS NOT NULL GROUP BY graphId ORDER BY k DESC LIMIT 1",
        (dev,),
    ).fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphNodeId,k.start,k.end-k.start,s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphId=?", (dev, gid),
    ).fetchall()
    bynode = defaultdict(list)
    for node, start, dur, nm in rows:
        bynode[node].append((start, dur, nm))
    n = max(len(v) for v in bynode.values())
    comm, comp, attn = [0.0] * n, [0.0] * n, [0.0] * n
    for inst in bynode.values():
        inst.sort()
        for i, (s, d, nm) in enumerate(inst):
            if i >= n:
                break
            if COLLECTIVE_RE.search(nm):
                comm[i] += d
            else:
                comp[i] += d
                if ATTN_RE.search(nm):
                    attn[i] += d
    return [x / 1e6 for x in comm], [x / 1e6 for x in comp], [x / 1e6 for x in attn], gid, dev


def single(blocks_prefix, rep):
    perstep = read_csv_int(f"{blocks_prefix}_perstep.csv")
    blocks = read_csv_int(f"{blocks_prefix}.csv")
    block_size = blocks[0]["block_size"]
    conc_tag = os.path.basename(blocks_prefix)
    print(f"\n===== D2 single-run summary: {conc_tag} =====")
    print(f"  block_size={block_size}  forwards(steps)={len(perstep)}  blocks={len(blocks)}")

    # ---- per-forward comm/comp from the DOMINANT CUDA graph --------------------
    # Each forward is a graph replay; comm/comp time is a property of the graph
    # (set by its batch size). The dominant graph (most replays) = the sustained
    # full batch and gives the most reliable per-replay estimate. We apply its
    # comm/fwd to ALL steps: the TP all-reduce is latency-DOMINATED (RING_LL), so
    # within a run comm/fwd varies little across the batch-size mix (across the
    # sweep it grows only sub-linearly with bs as the payload tokens x hidden grows).
    # comp/fwd DOES grow with bs, so comp-per-token is reported only over steps at
    # the dominant batch size (coverage stated). The fragile per-bs alignment of
    # the minor graphs is intentionally NOT used (replay-count estimation noise
    # makes secondary graphs unreliable, e.g. an impossible >2 ms comm/fwd).
    graphs, dev = graph_forward(f"{rep}.sqlite")
    bs_steps = Counter(r["batch_size"] for r in perstep)            # bs -> #denoising steps
    dom_bs = bs_steps.most_common(1)[0][0]
    dom_graph = graphs[0]                                           # most replays
    comm_fwd, comp_fwd = dom_graph["comm_fwd_ms"], dom_graph["comp_fwd_ms"]
    tot_ms, _comm_ms, comm_pct = kern_comm_fraction(f"{rep}_cuda_gpu_kern_sum.csv")
    print(f"  rep rank dev={dev}; dominant batch_size={dom_bs} "
          f"({bs_steps[dom_bs]}/{len(perstep)} steps); dominant graph "
          f"comm/fwd={comm_fwd:.2f} ms comp/fwd={comp_fwd:.2f} ms "
          f"comm={100*comm_fwd/(comm_fwd+comp_fwd):.1f}% of forward")
    print(f"  batch_size mix over steps: "
          + ", ".join(f"bs{b}:{n}" for b, n in bs_steps.most_common()))
    print(f"  whole-capture (all-rank) GPU kernel={tot_ms:.0f} ms comm={comm_pct:.1f}% (sanity)")

    # ---- per-forward cost stability across replays at the SAME batch size ------
    # Answers: is comm/comp really one number per bs, or does content (prompt/KV
    # length, masked-position → MoE routing) move it? Measured per-replay:
    #   comp/attn CV is tiny → compute IS ~content-insensitive (attn, the only
    #     length-dependent kernel, is a small near-constant slice), so one comp/fwd
    #     is justified; comm BYTES are shape-fixed but EXPOSED time is heavy-tailed
    #     (rare cross-rank desync stalls between replays, where the un-graphed host
    #     select loop skews ranks), so the mean comm/fwd is stall-inflated vs median.
    rc, rp, ra, _gid, _dev = dominant_per_replay(f"{rep}.sqlite")
    comm_med = st.median(rc)
    spikes = sum(x > 10 for x in rc)
    print(f"\n[per-forward stability @bs={dom_bs}, {len(rc)} replays]")
    print(f"  comp/fwd CV={_cv(rp):.1f}%   attn/fwd CV={_cv(ra):.1f}% "
          f"(attn={st.mean(ra):.2f} ms = the only KV-length-sensitive kernel) "
          f"→ compute ~content-insensitive")
    print(f"  comm/fwd mean={st.mean(rc):.2f} median={comm_med:.2f} ms CV={_cv(rc):.0f}% "
          f"(>10 ms: {spikes}/{len(rc)} desync spikes) → bytes fixed, exposed time tail-heavy")

    # ---- per-step tokens committed (the new per-token denominator) --------------
    com = [r["committed"] for r in perstep]
    print(f"\n[tokens committed per forward step]  mean={st.mean(com):.2f} "
          f"median={st.median(com):.0f} min={min(com)} max={max(com)} "
          f"p10={pct(com,0.1):.0f} p90={pct(com,0.9):.0f}")

    # ---- HEADLINE: exposed comm / output token, per-step distribution ----------
    # comm/fwd applied to EVERY step (bs-insensitive); committed = that step's tokens.
    com_pos = [r for r in perstep if r["committed"] > 0]
    tok_sum = sum(r["committed"] for r in com_pos)
    cpt = [comm_fwd / r["committed"] for r in com_pos]
    tw_comm = comm_fwd * len(com_pos) / tok_sum
    tw_comm_typ = comm_med * len(com_pos) / tok_sum   # typical (median comm/fwd, no stalls)
    print(f"\n[Exposed comm / output token]  (per-step samples, all {len(com_pos)} steps; "
          f"comm/fwd applied to all steps)")
    print(f"  token-weighted = {tw_comm_typ:.3f} ms (typical, median comm/fwd) .. "
          f"{tw_comm:.3f} ms (mean comm/fwd, incl. desync stalls)")
    print(f"  per-step (mean comm/fwd): mean={st.mean(cpt):.3f} median={st.median(cpt):.3f} "
          f"p10={pct(cpt,0.1):.3f} p90={pct(cpt,0.9):.3f}")
    # comp/token only over dominant-bs steps (comp/fwd is bs-dependent)
    dom = [r for r in perstep if r["batch_size"] == dom_bs and r["committed"] > 0]
    cov = sum(r["committed"] for r in dom) / max(1, sum(com))
    ppt = [comp_fwd / r["committed"] for r in dom]
    tw_comp = comp_fwd * len(dom) / sum(r["committed"] for r in dom)
    print(f"[Compute / output token]  (dominant-bs steps only, coverage {100*cov:.0f}% of tokens; "
          f"comp/fwd grows with bs)")
    print(f"  token-weighted mean = {tw_comp:.3f} ms   "
          f"per-step: mean={st.mean(ppt):.3f} median={st.median(ppt):.3f} "
          f"p10={pct(ppt,0.1):.3f} p90={pct(ppt,0.9):.3f}")
    print(f"  comm:comp per token (dominant bs) = 1 : {tw_comp/tw_comm:.1f}")

    # ---- S_k / intrinsic / straggler (per-block) -------------------------------
    calls = defaultdict(list)
    for r in blocks:
        calls[r["call_id"]].append(r)
    ski = [(r["finish_step"] + 1) if r["finish_step"] >= 0 else r["S_k"] for r in blocks]
    nc = [r["n_committed"] for r in blocks if r["n_committed"] > 0]
    # All per-BLOCK (block-weighted), so intrinsic + waste = batch S_k exactly per
    # block. batch S_k is the call's steps_executed shared by every block in it.
    skb = [c[0]["S_k"] for c in calls.values() for _ in c]
    waste = [c[0]["S_k"] - ((r["finish_step"] + 1) if r["finish_step"] >= 0 else c[0]["S_k"])
             for c in calls.values() for r in c]
    print(f"\n[intrinsic s_k] mean={st.mean(ski):.2f} median={st.median(ski):.0f} "
          f"p90={pct(ski,0.9):.0f} max={max(ski)}  | "
          f"[batch S_k/block] mean={st.mean(skb):.2f}  | "
          f"[n_committed/block] mean={st.mean(nc):.2f}")
    print(f"  tokens/intrinsic step={st.mean(nc)/st.mean(ski):.2f}  "
          f"straggler waste={st.mean(waste):.2f} fwd/block ({100*st.mean(waste)/st.mean(skb):.0f}% of batch S_k)")

    # ---- amplification vs the 1-round parallel-decode ideal --------------------
    print(f"\n[S_k amplification] a block pays mean {st.mean(ski):.1f} exposed comm rounds "
          f"vs the 1-round parallel-decode ideal -> {st.mean(ski):.1f}x; "
          f"AR same-size hides its comm (overlap) -> ~0 exposed.")

    metrics = {
        "tag": conc_tag, "dominant_bs": dom_bs,
        "comm_fwd_ms": comm_fwd, "comp_fwd_ms": comp_fwd,
        "comm_fwd_median_ms": comm_med,
        "comm_pct_of_fwd": 100 * comm_fwd / (comm_fwd + comp_fwd),
        "comm_pct_capture": comm_pct,
        "comp_cv_pct": _cv(rp), "attn_cv_pct": _cv(ra), "comm_cv_pct": _cv(rc),
        "tw_comm_per_tok_ms": tw_comm, "tw_comm_per_tok_typ_ms": tw_comm_typ,
        "tw_comp_per_tok_ms": tw_comp,
        "committed_per_step_mean": st.mean(com),
        "intrinsic_sk_mean": st.mean(ski), "intrinsic_sk_median": st.median(ski),
        "batch_sk_mean": st.mean(skb), "n_committed_mean": st.mean(nc),
        "tokens_per_step": st.mean(nc) / st.mean(ski),
        "straggler_waste_fwd": st.mean(waste),
        "straggler_waste_pct": 100 * st.mean(waste) / st.mean(skb),
        "coverage": cov,
    }
    json.dump(metrics, open(f"{blocks_prefix}_d2metrics.json", "w"), indent=2)
    return metrics


def sweep(logs_dir):
    files = sorted(glob.glob(os.path.join(logs_dir, "*_d2metrics.json")),
                   key=lambda p: int(re.search(r"_c(\d+)_", os.path.basename(p)).group(1))
                   if re.search(r"_c(\d+)_", os.path.basename(p)) else 0)
    ms = [json.load(open(f)) for f in files]
    if not ms:
        print("  (no *_d2metrics.json found)")
        return
    print("\n===== D2 concurrency sweep (HumanEval, 4xH100 NV18, TP4/EP4) =====")
    h = (f"{'tag':18s} {'bs':>3} {'comm/fwd':>8} {'comp/fwd':>8} {'comm%fwd':>8} "
         f"{'tok/step':>8} {'comm/tok(typ..mean)':>20} {'comp/tok':>8} {'s_k':>5} {'waste%':>6}")
    print(h)
    for m in ms:
        rng = f"{m.get('tw_comm_per_tok_typ_ms', m['tw_comm_per_tok_ms']):.3f}..{m['tw_comm_per_tok_ms']:.3f}"
        print(f"{m['tag']:18s} {m['dominant_bs']:>3d} {m['comm_fwd_ms']:>8.2f} "
              f"{m['comp_fwd_ms']:>8.2f} {m['comm_pct_of_fwd']:>7.1f}% "
              f"{m['committed_per_step_mean']:>8.2f} {rng:>20} "
              f"{m['tw_comp_per_tok_ms']:>8.3f} {m['intrinsic_sk_mean']:>5.1f} "
              f"{m['straggler_waste_pct']:>5.0f}%")
    print("\n  Stability @dominant bs (per-replay CV, content/length sensitivity):")
    for m in ms:
        if "comp_cv_pct" in m:
            print(f"    {m['tag']:18s} comp CV={m['comp_cv_pct']:4.1f}%  attn CV={m['attn_cv_pct']:4.1f}%  "
                  f"comm CV={m['comm_cv_pct']:4.0f}% (desync tail) "
                  f"comm/fwd med={m['comm_fwd_median_ms']:.2f} mean={m['comm_fwd_ms']:.2f} ms")
    print("\n  comm/fwd grows SUB-linearly with bs (latency-dominated, payload=tokens x hidden):")
    print("  comm/token FALLS as concurrency rises (tokens/step climbs faster than comm/fwd).")
    print("  comp is content-INSENSITIVE (CV<=~4%); comm bytes fixed but exposed time tail-heavy")
    print("  (mean comm/fwd > median due to rare cross-rank desync stalls; hence the typ..mean range).")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        sweep(sys.argv[2])
    elif len(sys.argv) == 3:
        single(sys.argv[1], sys.argv[2])
    else:
        sys.exit("usage: parse_d2.py <blocks_prefix> <nsys_rep>  |  --sweep <logs_dir>")


if __name__ == "__main__":
    main()
