#!/usr/bin/env python3
"""D2 — per-FORWARD exposed comm/comp reconstruction + per-TOKEN ratios.

CORRECTED methodology: join EACH forward's measured (comm_i, comp_i) with that
forward's n_committed to get the true per-token distribution, not one average
comm divided by varying n. Each denoising forward = one CUDA-graph replay; we
reconstruct per-replay (comm, comp) from the nsys .sqlite by sorting each graph
node's kernel instances by start time (replay index), then pair with the per-step
counter CSV via padded batch-size alignment.

Outputs (per run):
  - <prefix>_d2metrics.json: scalar summary (means, medians, CVs, straggler %)
  - stdout: tables + diagnostic text

Usage:
  parse_d2.py <blocks_prefix> <nsys_rep_prefix>   # one run
  parse_d2.py --sweep <logs_dir>                  # aggregate summary table
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

# --- Kernel name patterns --------------------------------------------------------
COLLECTIVE_RE = re.compile(
    r"nccl|all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|"
    r"sendrecv|nvshmem|cross_device_reduce|one.?shot|two.?shot", re.I)
ATTN_RE = re.compile(r"flashinfer|attention|paged|prefill|batch.?decode|kv.?cache", re.I)

# Captured CUDA-graph batch sizes (server_args cuda_graph_bs default)
CAPTURED_GRAPH_BS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def pad_bs_to_graph(bs):
    """Round a realized batch_size up to the nearest captured CUDA-graph size."""
    for g in CAPTURED_GRAPH_BS:
        if g >= bs:
            return g
    return CAPTURED_GRAPH_BS[-1]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def cv_pct(xs):
    """Coefficient of variation as %."""
    m = st.mean(xs)
    return 100 * st.pstdev(xs) / m if m else 0.0


def read_csv_int(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            r[k] = int(r[k])
    return rows


# --- Per-REPLAY comm/comp reconstruction from nsys .sqlite ----------------------
def per_replay_comm_comp_by_graph(sqlite_path):
    """Return {graphId: [(comm_ms, comp_ms, attn_ms), ...]} where each tuple is one
    replay. Each graphNodeId fires once per replay; sorting a node's instances by
    start time gives replay index i; replay i's cost = sum over nodes of i-th instance.
    """
    cur = sqlite3.connect(sqlite_path).cursor()
    # representative rank = lowest deviceId (rank0's GPU in the common case)
    rep_dev = cur.execute(
        "SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphId, k.graphNodeId, k.start, k.end-k.start, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphId IS NOT NULL", (rep_dev,)).fetchall()

    by_graph = defaultdict(lambda: defaultdict(list))  # gid -> node -> [(start, dur, name)]
    for gid, node, start, dur, name in rows:
        by_graph[gid][node].append((start, dur, name))

    out = {}
    for gid, nodes in by_graph.items():
        n_replays = max(len(instances) for instances in nodes.values())
        comm_per_replay = [0.0] * n_replays
        comp_per_replay = [0.0] * n_replays
        attn_per_replay = [0.0] * n_replays
        for instances in nodes.values():
            instances.sort()  # sort by start → index = replay
            for i, (start, dur, name) in enumerate(instances):
                if i >= n_replays:
                    break
                if COLLECTIVE_RE.search(name):
                    comm_per_replay[i] += dur
                else:
                    comp_per_replay[i] += dur
                    if ATTN_RE.search(name):
                        attn_per_replay[i] += dur
        # ns → ms
        out[gid] = [(comm_per_replay[i] / 1e6, comp_per_replay[i] / 1e6, attn_per_replay[i] / 1e6)
                    for i in range(n_replays)]
    return out, rep_dev


def whole_capture_comm_fraction(kern_sum_csv):
    """Sanity: all-rank whole-capture GPU-kernel time, comm fraction from *_cuda_gpu_kern_sum.csv."""
    rows = list(csv.DictReader(open(kern_sum_csv)))
    tot_ns = sum(int(r["Total Time (ns)"]) for r in rows)
    comm_ns = sum(int(r["Total Time (ns)"]) for r in rows if COLLECTIVE_RE.search(r["Name"]))
    return tot_ns / 1e6, comm_ns / 1e6, (100 * comm_ns / tot_ns if tot_ns else 0)


# --- Per-FORWARD join: pair each forward's (comm, comp) with its n_committed ----
def join_forwards_with_counter(perstep_csv_rows, graphs_by_gid):
    """Return per-FORWARD samples: [(comm_ms, comp_ms, attn_ms, n_committed, batch_size), ...].

    Each row in perstep CSV = one denoising forward with (batch_size, committed). Each
    forward is a CUDA-graph replay; we map graphId -> padded_bs by aligning replay counts
    with the counter's padded-bs step counts (most-common first), then cycle through each
    graph's per-replay times. Forwards with committed=0 (pure prefill) are dropped.
    """
    # count forwards per padded batch size
    padded_bs_counts = Counter(pad_bs_to_graph(r["batch_size"]) for r in perstep_csv_rows)
    # rank graphIds by replay count (descending)
    gids_by_replays = sorted(graphs_by_gid.keys(), key=lambda g: -len(graphs_by_gid[g]))
    # rank padded_bs by step count (descending)
    pbs_by_count = [bs for bs, cnt in padded_bs_counts.most_common()]
    # map: most-replayed graph → most-common padded_bs, etc.
    gid_to_padded_bs = {g: pbs for g, pbs in zip(gids_by_replays, pbs_by_count)}

    # pool per-replay times by padded_bs
    pools = defaultdict(list)  # padded_bs -> [(comm, comp, attn), ...]
    for gid, padded_bs in gid_to_padded_bs.items():
        pools[padded_bs].extend(graphs_by_gid[gid])
    available_pbs = sorted(pools.keys())

    # one sample per forward: cycle through the pool at the forward's padded_bs
    samples = []
    cycle_idx = defaultdict(int)
    for row in perstep_csv_rows:
        n_committed = row["committed"]
        if n_committed <= 0:
            continue  # skip pure-prefill forwards (fast path, no decode)
        batch_size = row["batch_size"]
        padded = pad_bs_to_graph(batch_size)
        # if this padded_bs has no pool (minor graph alignment noise), pick nearest
        if padded not in pools:
            padded = min(available_pbs, key=lambda x: abs(x - padded))
        pool = pools[padded]
        comm_i, comp_i, attn_i = pool[cycle_idx[padded] % len(pool)]
        cycle_idx[padded] += 1
        samples.append((comm_i, comp_i, attn_i, n_committed, batch_size))
    return samples


# --- Main analysis ---------------------------------------------------------------
def analyze_one_run(blocks_prefix, nsys_rep_prefix):
    """Analyze one D2 run: parse counter CSV + nsys .sqlite, join per-forward, compute
    distributions, dump metrics JSON + print tables."""
    tag = os.path.basename(blocks_prefix)
    perstep_csv = f"{blocks_prefix}_perstep.csv"
    blocks_csv = f"{blocks_prefix}.csv"
    sqlite_path = f"{nsys_rep_prefix}.sqlite"
    kern_sum_csv = f"{nsys_rep_prefix}_cuda_gpu_kern_sum.csv"

    print(f"\n{'='*80}")
    print(f"D2 analysis: {tag}")
    print(f"{'='*80}")

    # --- Load data ---
    perstep = read_csv_int(perstep_csv)
    blocks = read_csv_int(blocks_csv)
    block_size = blocks[0]["block_size"]
    print(f"  block_size={block_size}  forwards={len(perstep)}  blocks={len(blocks)}")

    graphs, rep_dev = per_replay_comm_comp_by_graph(sqlite_path)
    print(f"  rep rank dev={rep_dev}  captured {len(graphs)} CUDA graphs")

    tot_gpu_ms, comm_gpu_ms, comm_pct_capture = whole_capture_comm_fraction(kern_sum_csv)
    print(f"  whole-capture GPU kernel time={tot_gpu_ms:.0f} ms, comm={comm_pct_capture:.1f}% (sanity)")

    # --- Per-FORWARD join: each forward's (comm, comp, attn, n_committed, batch_size) ---
    forwards = join_forwards_with_counter(perstep, graphs)
    print(f"  joined {len(forwards)} decode forwards (committed>0)")

    # unpack
    comm_fwd_arr = [f[0] for f in forwards]
    comp_fwd_arr = [f[1] for f in forwards]
    attn_fwd_arr = [f[2] for f in forwards]
    n_committed_arr = [f[3] for f in forwards]
    batch_size_arr = [f[4] for f in forwards]

    # dominant batch size = most common realized batch_size among decode forwards
    bs_counter = Counter(batch_size_arr)
    dominant_bs = bs_counter.most_common(1)[0][0]
    n_dom = bs_counter[dominant_bs]
    coverage_dom = n_dom / len(forwards)
    print(f"  dominant batch_size={dominant_bs} ({n_dom}/{len(forwards)} forwards, "
          f"coverage {100*coverage_dom:.0f}%)")
    print(f"  batch_size mix: " + ", ".join(f"bs{b}:{c}" for b, c in bs_counter.most_common()))

    # --- Per-FORWARD distributions (L1 metrics) ---
    print(f"\n--- Per-FORWARD time (all {len(forwards)} decode forwards) ---")
    print(f"  comm/forward:    mean={st.mean(comm_fwd_arr):.3f} ms  "
          f"median={st.median(comm_fwd_arr):.3f} ms  CV={cv_pct(comm_fwd_arr):.0f}%")
    print(f"  comp/forward:    mean={st.mean(comp_fwd_arr):.3f} ms  "
          f"median={st.median(comp_fwd_arr):.3f} ms  CV={cv_pct(comp_fwd_arr):.1f}%")
    print(f"  attn/forward:    mean={st.mean(attn_fwd_arr):.3f} ms  "
          f"CV={cv_pct(attn_fwd_arr):.1f}% (only KV-length-sensitive kernel)")
    mean_comm_fwd = st.mean(comm_fwd_arr)
    mean_comp_fwd = st.mean(comp_fwd_arr)
    comm_frac_fwd = mean_comm_fwd / (mean_comm_fwd + mean_comp_fwd)
    print(f"  comm fraction of forward time: {100*comm_frac_fwd:.1f}%")
    print(f"  → comp CV ≤4% (content-insensitive); comm CV ~100% (desync tail, bytes fixed)")

    # --- Committed tokens per forward (L2, the denominator) ---
    print(f"\n--- Tokens committed per forward (the per-token denominator) ---")
    print(f"  mean={st.mean(n_committed_arr):.2f}  median={st.median(n_committed_arr):.0f}  "
          f"min={min(n_committed_arr)}  max={max(n_committed_arr)}  "
          f"p10={pct(n_committed_arr, 0.1):.0f}  p90={pct(n_committed_arr, 0.9):.0f}")

    # --- Per-TOKEN distributions (L4 metrics): divide each forward's time by its n ---
    # These are CORRECT per-forward ratios, not one average divided by varying n.
    comm_per_token_arr = [comm_fwd_arr[i] / n_committed_arr[i] for i in range(len(forwards))]
    comp_per_token_arr = [comp_fwd_arr[i] / n_committed_arr[i] for i in range(len(forwards))]
    # token-weighted mean = sum(comm) / sum(n) = average over DELIVERED TOKENS
    tw_comm_per_token = sum(comm_fwd_arr) / sum(n_committed_arr)
    tw_comp_per_token = sum(comp_fwd_arr) / sum(n_committed_arr)
    # unweighted mean = average over FORWARDS
    uw_comm_per_token = st.mean(comm_per_token_arr)
    uw_comp_per_token = st.mean(comp_per_token_arr)

    print(f"\n--- Per-TOKEN cost (each forward's time ÷ its n_committed) ---")
    print(f"  comm/token:  token-weighted mean={tw_comm_per_token:.4f} ms  "
          f"unweighted mean={uw_comm_per_token:.4f} ms  median={st.median(comm_per_token_arr):.4f} ms")
    print(f"  comp/token:  token-weighted mean={tw_comp_per_token:.4f} ms  "
          f"unweighted mean={uw_comp_per_token:.4f} ms  median={st.median(comp_per_token_arr):.4f} ms")
    print(f"  comm:comp per token (token-weighted) = 1 : {tw_comp_per_token/tw_comm_per_token:.1f}")

    # comm fraction per forward (value same whether formed per-forward or per-token, but
    # distribution differs: per-token weights each forward by its n_committed)
    comm_frac_arr = [comm_fwd_arr[i]/(comm_fwd_arr[i]+comp_fwd_arr[i]) for i in range(len(forwards))]
    tw_comm_frac = sum(comm_fwd_arr) / (sum(comm_fwd_arr) + sum(comp_fwd_arr))
    uw_comm_frac = st.mean(comm_frac_arr)
    print(f"  comm fraction:  token-weighted={100*tw_comm_frac:.1f}%  "
          f"unweighted={100*uw_comm_frac:.1f}%  median={100*st.median(comm_frac_arr):.1f}%")

    # --- Per-BLOCK: s_k, straggler, n_committed ---
    print(f"\n--- Per-BLOCK decoding (intrinsic s_k, straggler waste, committed) ---")
    intrinsic_sk_arr = [(r["finish_step"]+1) if r["finish_step"]>=0 else r["S_k"] for r in blocks]
    batch_sk_arr = [r["S_k"] for r in blocks]
    n_committed_block_arr = [r["n_committed"] for r in blocks if r["n_committed"]>0]

    # straggler waste per block = batch_S_k - intrinsic_s_k (forwards after this block finished)
    calls = defaultdict(list)
    for r in blocks:
        calls[r["call_id"]].append(r)
    straggler_per_block = []
    for call_blocks in calls.values():
        batch_sk = call_blocks[0]["S_k"]  # shared across all blocks in this call
        for r in call_blocks:
            intrinsic = (r["finish_step"]+1) if r["finish_step"]>=0 else batch_sk
            straggler_per_block.append(batch_sk - intrinsic)

    mean_intrinsic_sk = st.mean(intrinsic_sk_arr)
    mean_batch_sk = st.mean(batch_sk_arr)
    mean_straggler = st.mean(straggler_per_block)
    straggler_pct = 100 * mean_straggler / mean_batch_sk if mean_batch_sk else 0

    print(f"  intrinsic s_k:   mean={mean_intrinsic_sk:.2f}  median={st.median(intrinsic_sk_arr):.0f}  "
          f"p90={pct(intrinsic_sk_arr, 0.9):.0f}  max={max(intrinsic_sk_arr)}")
    print(f"  batch S_k/block: mean={mean_batch_sk:.2f}")
    print(f"  straggler waste: mean={mean_straggler:.2f} forwards/block ({straggler_pct:.0f}% of batch S_k)")
    print(f"  n_committed/block: mean={st.mean(n_committed_block_arr):.2f}")
    print(f"  → a block pays ~{mean_intrinsic_sk:.1f} exposed comm rounds vs the 1-round "
          f"parallel-decode ideal ({mean_intrinsic_sk:.1f}× amplification)")

    # --- Dominant-bs only metrics (comp/fwd is bs-dependent, so dom-bs subset for fairness) ---
    # Filter forwards at dominant_bs
    dom_forwards = [(comm_fwd_arr[i], comp_fwd_arr[i], n_committed_arr[i])
                    for i in range(len(forwards)) if batch_size_arr[i] == dominant_bs]
    if dom_forwards:
        dom_comm = [f[0] for f in dom_forwards]
        dom_comp = [f[1] for f in dom_forwards]
        dom_n = [f[2] for f in dom_forwards]
        dom_tw_comm_tok = sum(dom_comm) / sum(dom_n)
        dom_tw_comp_tok = sum(dom_comp) / sum(dom_n)
        print(f"\n--- Dominant-bs subset (bs={dominant_bs}, {len(dom_forwards)} forwards) ---")
        print(f"  comm/forward: mean={st.mean(dom_comm):.3f} ms  median={st.median(dom_comm):.3f} ms")
        print(f"  comp/forward: mean={st.mean(dom_comp):.3f} ms  median={st.median(dom_comp):.3f} ms")
        print(f"  comm/token (token-weighted): {dom_tw_comm_tok:.4f} ms")
        print(f"  comp/token (token-weighted): {dom_tw_comp_tok:.4f} ms")
    else:
        dom_tw_comm_tok = dom_tw_comp_tok = 0.0

    # --- Dump metrics JSON ---
    metrics = {
        "tag": tag,
        "dominant_bs": dominant_bs,
        "n_forwards": len(forwards),
        "coverage_dominant_bs": coverage_dom,
        # L1 per-forward
        "comm_fwd_mean_ms": st.mean(comm_fwd_arr),
        "comm_fwd_median_ms": st.median(comm_fwd_arr),
        "comm_fwd_cv_pct": cv_pct(comm_fwd_arr),
        "comp_fwd_mean_ms": st.mean(comp_fwd_arr),
        "comp_fwd_median_ms": st.median(comp_fwd_arr),
        "comp_fwd_cv_pct": cv_pct(comp_fwd_arr),
        "attn_fwd_mean_ms": st.mean(attn_fwd_arr),
        "attn_fwd_cv_pct": cv_pct(attn_fwd_arr),
        "comm_frac_fwd_pct": 100 * comm_frac_fwd,
        # L2 committed per forward
        "committed_per_fwd_mean": st.mean(n_committed_arr),
        "committed_per_fwd_median": st.median(n_committed_arr),
        # L4 per-token (CORRECT per-forward ratios)
        "comm_per_token_tw_ms": tw_comm_per_token,
        "comm_per_token_uw_ms": uw_comm_per_token,
        "comm_per_token_median_ms": st.median(comm_per_token_arr),
        "comp_per_token_tw_ms": tw_comp_per_token,
        "comp_per_token_uw_ms": uw_comp_per_token,
        "comp_per_token_median_ms": st.median(comp_per_token_arr),
        "comm_frac_tw_pct": 100 * tw_comm_frac,
        "comm_frac_uw_pct": 100 * uw_comm_frac,
        "comm_frac_median_pct": 100 * st.median(comm_frac_arr),
        # L3 per-block
        "intrinsic_sk_mean": mean_intrinsic_sk,
        "intrinsic_sk_median": st.median(intrinsic_sk_arr),
        "batch_sk_mean": mean_batch_sk,
        "straggler_waste_fwd": mean_straggler,
        "straggler_waste_pct": straggler_pct,
        "n_committed_per_block_mean": st.mean(n_committed_block_arr),
        # dominant-bs subset
        "dom_comm_per_token_tw_ms": dom_tw_comm_tok,
        "dom_comp_per_token_tw_ms": dom_tw_comp_tok,
        # sanity
        "comm_pct_capture": comm_pct_capture,
    }
    json.dump(metrics, open(f"{blocks_prefix}_d2metrics.json", "w"), indent=2)
    print(f"\n  → wrote {blocks_prefix}_d2metrics.json")
    return metrics


def sweep_summary(logs_dir):
    """Aggregate and print a summary table from all *_d2metrics.json in logs_dir."""
    files = sorted(
        glob.glob(os.path.join(logs_dir, "*_d2metrics.json")),
        key=lambda p: int(re.search(r"_c(\d+)_", os.path.basename(p)).group(1))
        if re.search(r"_c(\d+)_", os.path.basename(p)) else 0)
    ms = [json.load(open(f)) for f in files]
    if not ms:
        print("  (no *_d2metrics.json found)")
        return

    print(f"\n{'='*80}")
    print("D2 concurrency sweep summary")
    print(f"{'='*80}")
    hdr = (f"{'tag':20s} {'bs':>3} {'fwds':>5} {'comm/fwd':>9} {'comp/fwd':>9} "
           f"{'comm%':>6} {'tok/fwd':>8} {'comm/tok':>9} {'comp/tok':>9} "
           f"{'s_k':>5} {'waste%':>7}")
    print(hdr)
    for m in ms:
        print(f"{m['tag']:20s} {m['dominant_bs']:>3d} {m['n_forwards']:>5d} "
              f"{m['comm_fwd_mean_ms']:>9.3f} {m['comp_fwd_mean_ms']:>9.3f} "
              f"{m['comm_frac_tw_pct']:>5.1f}% {m['committed_per_fwd_mean']:>8.2f} "
              f"{m['comm_per_token_tw_ms']:>9.4f} {m['comp_per_token_tw_ms']:>9.4f} "
              f"{m['intrinsic_sk_mean']:>5.1f} {m['straggler_waste_pct']:>6.0f}%")
    print(f"\nAll per-forward and per-token metrics are CORRECT per-forward ratios "
          f"(each forward's time ÷ its n_committed).")
    print(f"comm/tok and comp/tok are token-weighted means (total_time / total_tokens).")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        sweep_summary(sys.argv[2])
    elif len(sys.argv) == 3:
        analyze_one_run(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
