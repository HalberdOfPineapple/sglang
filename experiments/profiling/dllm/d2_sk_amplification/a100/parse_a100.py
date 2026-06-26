#!/usr/bin/env python3
"""D2 / A100 — PROJECTED-NVLink comm fraction per output token, from comm VOLUME.

Same per-step analysis as the H100 leaf (parse_d2.py), but the COMM time is PROJECTED
from analytic volume onto A100 NVLink while the COMPUTE time is the REAL A100
measurement -- because the raw PCIe comm time is interconnect-bound and meaningless to
compare against NVLink. Every denoising forward is a CUDA-graph replay; we reconstruct
its compute per replay from the nsys .sqlite and tie it to the tokens it committed
(the §3 per-step counter).

  1. MEASURED A100 compute/forward (interconnect-independent, content-insensitive).
     The raw PCIe comm time is reconstructed too but DISCARDED (sanity line only).
  2. comm VOLUME is analytic & deterministic. Per forward: 41 TP all-reduces of
     [bs*block, hidden] (msg M=bs*block*hidden*2B, ring bus = 2(N-1)/N*M each) + 1
     vocab all-gather of [bs*block, vocab] (gathered G=bs*block*vocab*2B, ring bus =
     (N-1)/N*G). Cross-checked: 41 all-reduce + 1 all-gather instances/replay.
  3. PROJECTED A100-NVLink comm/forward = total_bus_traffic / busbw  (busbw in
     BUSBW_REFS: A100 NVLink3 achievable 240 GB/s headline, 300 GB/s peak).
  4. COMM FRACTION = projected_comm / (projected_comm + measured_compute), and the
     per-token split -- the D2 metric, but projected. NOTE: volume/busbw is a
     bandwidth model and the 41 small all-reduces are partly latency-bound, so this
     is a LOWER bound (the H100 sibling MEASURES 22-32%).

Usage:
  parse_a100.py <blocks_prefix> <nsys_rep_prefix>   # one run
  parse_a100.py --sweep <logs_dir>                  # aggregate *_a100metrics.json
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

# ---- model / parallelism constants (LLaDA2.0-mini, confirmed from D1/D2) -----
NUM_LAYERS = 20
HIDDEN = 2048
BLOCK = 32
DTYPE_BYTES = 2          # bf16
VOCAB = 157184
TP = 4
A_AR = 2 * NUM_LAYERS + 1  # all-reduces per forward (attn-out + MoE-out per layer + 1)
RING_FACTOR = 2 * (TP - 1) / TP    # ring all-reduce bus traffic = this * message size
AG_FACTOR = (TP - 1) / TP          # ring all-gather bus traffic = this * gathered size

# A100 NVLink reference busbw for the comm-time PROJECTION (bytes/s). We PROJECT the
# communication time from the analytic volume (projected_time = bus_traffic / busbw)
# and combine it with the REAL A100-measured COMPUTE time, to get the comm/compute
# balance this A100 box WOULD have on NVLink (the raw PCIe comm time is interconnect-
# bound and discarded). A100 SXM NVLink3 = 12 links x 25 GB/s/dir = 300 GB/s/dir
# (600 GB/s/GPU bidirectional); the achievable 4-GPU all-reduce busbw (nccl-tests
# class) is ~240 GB/s, the per-link unidirectional ceiling ~300 GB/s.
BUSBW_REFS = {"a100_nvlink_achievable": 240e9, "a100_nvlink_peak": 300e9}
BUSBW_HEADLINE = "a100_nvlink_achievable"

COLLECTIVE_RE = re.compile(
    r"nccl|all.?reduce|all.?to.?all|all.?gather|reduce.?scatter|"
    r"sendrecv|nvshmem|cross_device_reduce|one.?shot|two.?shot", re.I)
ALLGATHER_RE = re.compile(r"all.?gather", re.I)
ATTN_RE = re.compile(r"flashinfer|attention|paged|prefill|batch.?decode|kv.?cache", re.I)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def read_csv_int(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            r[k] = int(r[k])
    return rows


def _cv(xs):
    m = st.mean(xs)
    return 100 * st.pstdev(xs) / m if m else 0.0


def msg_bytes(bs):
    """TP all-reduce payload (one call): [bs*block, hidden] activation, bf16."""
    return bs * BLOCK * HIDDEN * DTYPE_BYTES


def volume_fwd(bs, n_ar=A_AR):
    """(alg_volume, bus_traffic) per forward in bytes, for n_ar TP all-reduces."""
    m = msg_bytes(bs)
    return n_ar * m, n_ar * RING_FACTOR * m


def ag_volume_fwd(bs, n_ag=1):
    """Vocab LM-head all-gather: each rank holds [bs*block, vocab/TP] logits, gathered
    to [bs*block, vocab]. (gathered_bytes, ring bus_traffic) per forward.
    Unlike the small latency-bound all-reduces this is ONE large message, so it is
    bandwidth-bound — the volume/busbw projection is more faithful for it."""
    g = bs * BLOCK * VOCAB * DTYPE_BYTES
    return n_ag * g, n_ag * AG_FACTOR * g


# NOTE on nsys schema: the H100 sibling used nsys 2026.x whose KERNEL table has a
# `graphId` column; this A100 box's nsys is 2025.1.1 (bundled with nsight-compute)
# and has NO `graphId` — only `graphNodeId`. So we group graph replays on
# `graphNodeId` (present in every nsys version): every node of one captured CUDA
# graph fires exactly R times across the graph's R replays, so nodes bucketed by
# their replay count R reconstruct the graphs without a graphId. This is schema-
# independent (works on both the A100 and H100 .sqlite). Dominant graph = the
# bucket with the most TOTAL kernels (R x nodes) = the steady-state forward.
def _graph_nodes(sqlite_path):
    """In-graph kernels grouped by graphNodeId -> [(start,dur,name)], for the rep
    rank (min deviceId). graphNodeId NULL = eager (un-graphed host/select-loop)."""
    cur = sqlite3.connect(sqlite_path).cursor()
    dev = cur.execute("SELECT MIN(deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    rows = cur.execute(
        "SELECT k.graphNodeId,k.start,k.end-k.start,s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=? AND k.graphNodeId IS NOT NULL", (dev,)).fetchall()
    bynode = defaultdict(list)
    for node, start, dur, nm in rows:
        bynode[node].append((start, dur, nm))
    return bynode, dev


def _bucket_graphs(bynode):
    """node -> graph by replay count R (every node of a captured graph fires R times)."""
    graphs = defaultdict(list)
    for node, insts in bynode.items():
        graphs[len(insts)].append(node)
    return graphs


def graph_forward(sqlite_path):
    """Per CUDA-graph (replay-count bucket): replays + summed comm/comp/allreduce/
    allgather kernel ns and comm instance counts, for the rep rank."""
    bynode, dev = _graph_nodes(sqlite_path)
    out = []
    for R, nodes in _bucket_graphs(bynode).items():
        comm = comp = ag = ar = 0.0
        n_ar = n_ag = 0
        for node in nodes:
            for s, dur, nm in bynode[node]:
                if COLLECTIVE_RE.search(nm):
                    comm += dur
                    if ALLGATHER_RE.search(nm):
                        ag += dur; n_ag += 1
                    else:
                        ar += dur; n_ar += 1
                else:
                    comp += dur
        out.append({
            "gid": R, "replays": R, "nnodes": len(nodes),
            "comm_fwd_ms": comm / R / 1e6, "comp_fwd_ms": comp / R / 1e6,
            "ar_fwd_ms": ar / R / 1e6, "ag_fwd_ms": ag / R / 1e6,
            "n_ar": n_ar / R, "n_ag": n_ag / R,
        })
    out.sort(key=lambda x: -x["replays"] * x["nnodes"])
    return out, dev


def dominant_per_replay(sqlite_path):
    """Per-REPLAY comm/comp/attn/allreduce time (ms) for the dominant graph — measures
    whether per-forward cost is constant across forwards at the same bs."""
    bynode, dev = _graph_nodes(sqlite_path)
    graphs = _bucket_graphs(bynode)
    Rdom = max(graphs, key=lambda R: R * len(graphs[R]))   # heaviest = steady forward
    n = Rdom
    comm, comp, attn, ar = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    for node in graphs[Rdom]:
        for i, (s, d, nm) in enumerate(sorted(bynode[node])):
            if i >= n:
                break
            if COLLECTIVE_RE.search(nm):
                comm[i] += d
                if not ALLGATHER_RE.search(nm):
                    ar[i] += d
            else:
                comp[i] += d
                if ATTN_RE.search(nm):
                    attn[i] += d
    f = lambda a: [x / 1e6 for x in a]
    return f(comm), f(comp), f(attn), f(ar), Rdom, dev


def kern_comm_fraction(path):
    rows = list(csv.DictReader(open(path)))
    tot = sum(int(r["Total Time (ns)"]) for r in rows)
    comm = sum(int(r["Total Time (ns)"]) for r in rows if COLLECTIVE_RE.search(r["Name"]))
    return tot / 1e6, comm / 1e6, (100 * comm / tot if tot else 0)


def gbps(bytes_, sec):
    return bytes_ / sec / 1e9 if sec else 0.0


def single(blocks_prefix, rep):
    perstep = read_csv_int(f"{blocks_prefix}_perstep.csv")
    blocks = read_csv_int(f"{blocks_prefix}.csv")
    block_size = blocks[0]["block_size"]
    tag = os.path.basename(blocks_prefix)
    print(f"\n===== D2/A100-PCIe single-run summary: {tag} =====")
    print(f"  block_size={block_size}  forwards(steps)={len(perstep)}  blocks={len(blocks)}")

    graphs, dev = graph_forward(f"{rep}.sqlite")
    bs_steps = Counter(r["batch_size"] for r in perstep)
    dom_bs = bs_steps.most_common(1)[0][0]
    dg = graphs[0]
    comm_fwd, comp_fwd = dg["comm_fwd_ms"], dg["comp_fwd_ms"]
    ar_fwd, ag_fwd = dg["ar_fwd_ms"], dg["ag_fwd_ms"]
    n_ar = round(dg["n_ar"]) or A_AR
    tot_ms, _c, comm_pct = kern_comm_fraction(f"{rep}_cuda_gpu_kern_sum.csv")
    print(f"  rep rank dev={dev}; dominant batch_size={dom_bs} "
          f"({bs_steps[dom_bs]}/{len(perstep)} steps)")
    print(f"  dominant graph: comm/fwd={comm_fwd:.3f} ms "
          f"(allreduce={ar_fwd:.3f} + allgather={ag_fwd:.3f}) comp/fwd={comp_fwd:.3f} ms "
          f"comm={100*comm_fwd/(comm_fwd+comp_fwd):.1f}% of fwd")
    print(f"  collective instances/replay: allreduce={dg['n_ar']:.1f} (model {A_AR}) "
          f"allgather={dg['n_ag']:.1f}")
    print(f"  batch_size mix: " + ", ".join(f"bs{b}:{n}" for b, n in bs_steps.most_common()))

    # ---- MEASURED A100 compute/fwd (interconnect-INDEPENDENT — the REAL part we keep)
    # Per CUDA-graph replay; compute is content-insensitive (low CV) so the median is a
    # clean per-bs number. The raw PCIe comm time is ALSO here but we DISCARD it as the
    # headline (it is interconnect-bound, not the dLLM's intrinsic comm cost — comparing
    # it to NVLink is meaningless); we project comm from volume instead (below).
    rc, rp, ra, rar, _gid, _dev = dominant_per_replay(f"{rep}.sqlite")
    comp_fwd = st.median(rp)             # MEASURED A100 compute time/fwd (ms)
    pcie_comm_meas = st.median(rc)       # raw PCIe comm/fwd (sanity only, NOT used)
    print(f"\n[per-forward, dominant graph @bs={dom_bs}, {len(rp)} replays]")
    print(f"  MEASURED A100 compute/fwd = {comp_fwd:.3f} ms "
          f"(CV {_cv(rp):.1f}%, attn CV {_cv(ra):.1f}% — content-insensitive)")
    print(f"  (raw A100/PCIe comm/fwd = {pcie_comm_meas:.1f} ms — interconnect-bound, DISCARDED; "
          f"comm is projected from volume below)")

    # ---- comm VOLUME (analytic) -> PROJECTED A100-NVLink comm time -------------
    # The dLLM's intrinsic per-forward comm is fixed by the model+bs (shape-deterministic):
    # 41 TP all-reduces of [bs*block,hidden] + 1 vocab all-gather of [bs*block,vocab]. We
    # PROJECT its time on A100 NVLink as bus_traffic / busbw, then form the comm fraction
    # against the MEASURED A100 compute — i.e. the comm/compute balance this box WOULD
    # have on NVLink. (Volume/busbw is a bandwidth model; see caveat — the many small
    # all-reduces are partly latency-bound so this projection is a lower bound on comm.)
    alg_v, bus_v = volume_fwd(dom_bs, n_ar)
    ag_g, ag_bus = ag_volume_fwd(dom_bs)
    mB = msg_bytes(dom_bs)
    bw = BUSBW_REFS[BUSBW_HEADLINE]
    proj_ar = bus_v / bw * 1e3
    proj_ag = ag_bus / bw * 1e3
    proj_comm = proj_ar + proj_ag                      # projected total comm/fwd (ms)
    tot_bus = bus_v + ag_bus
    comm_frac = 100 * proj_comm / (proj_comm + comp_fwd)
    print(f"\n[comm VOLUME @bs={dom_bs}]  all-reduce {n_ar}x{mB/1e6:.3f}MB -> bus {bus_v/1e6:.1f}MB ; "
          f"vocab all-gather {ag_g/1e6:.1f}MB -> bus {ag_bus/1e6:.1f}MB ; total bus {tot_bus/1e6:.1f}MB")
    print(f"[PROJECTED A100-NVLink comm/fwd @ {bw/1e9:.0f} GB/s]  "
          f"all-reduce={proj_ar:.3f} + all-gather={proj_ag:.3f} = {proj_comm:.3f} ms")
    print(f"[COMM FRACTION (projected)]  proj_comm/(proj_comm+measured_comp) "
          f"= {proj_comm:.3f}/({proj_comm:.3f}+{comp_fwd:.3f}) = {comm_frac:.1f}%")
    for name, b in BUSBW_REFS.items():                 # bandwidth sensitivity
        pc = tot_bus / b * 1e3
        print(f"    @ {name} ({b/1e9:.0f} GB/s): comm/fwd={pc:.3f} ms  frac={100*pc/(pc+comp_fwd):.1f}%")

    # ---- per-step tokens committed (the per-token denominator) -----------------
    com = [r["committed"] for r in perstep]
    print(f"\n[tokens committed per forward]  mean={st.mean(com):.2f} "
          f"median={st.median(com):.0f} min={min(com)} max={max(com)} "
          f"p10={pct(com,0.1):.0f} p90={pct(com,0.9):.0f}")

    # ---- per OUTPUT token: PROJECTED comm vs MEASURED compute (token-weighted) --
    com_pos = [r for r in perstep if r["committed"] > 0]
    tok_sum = sum(r["committed"] for r in com_pos)
    scale = len(com_pos) / tok_sum                     # forwards / delivered tokens
    tw_comm = proj_comm * scale                         # projected NVLink comm / token
    tw_vol = tot_bus * scale                            # bus bytes / delivered token
    dom = [r for r in perstep if r["batch_size"] == dom_bs and r["committed"] > 0]
    cov = sum(r["committed"] for r in dom) / max(1, sum(com))
    tw_comp = comp_fwd * len(dom) / sum(r["committed"] for r in dom) if dom else 0
    print(f"\n[per OUTPUT token, token-weighted]")
    print(f"  projected comm/tok = {tw_comm:.4f} ms   measured compute/tok = {tw_comp:.3f} ms "
          f"(dom-bs, cov {100*cov:.0f}%)   comm:comp = 1 : {tw_comp/tw_comm:.0f}")
    print(f"  volume {tw_vol/1e6:.2f} MB bus / delivered token")

    # ---- S_k / straggler (per-block) -------------------------------------------
    calls = defaultdict(list)
    for r in blocks:
        calls[r["call_id"]].append(r)
    ski = [(r["finish_step"] + 1) if r["finish_step"] >= 0 else r["S_k"] for r in blocks]
    nc = [r["n_committed"] for r in blocks if r["n_committed"] > 0]
    skb = [c[0]["S_k"] for c in calls.values() for _ in c]
    waste = [c[0]["S_k"] - ((r["finish_step"] + 1) if r["finish_step"] >= 0 else c[0]["S_k"])
             for c in calls.values() for r in c]
    print(f"\n[intrinsic s_k] mean={st.mean(ski):.2f} median={st.median(ski):.0f} "
          f"max={max(ski)} | [batch S_k/block] mean={st.mean(skb):.2f} | "
          f"straggler waste={100*st.mean(waste)/st.mean(skb):.0f}%")

    metrics = {
        "tag": tag, "dominant_bs": dom_bs, "n_ar": n_ar,
        "busbw_ref": BUSBW_HEADLINE, "busbw_gbps": bw / 1e9,
        # measured A100 compute (the real, interconnect-independent part)
        "comp_fwd_ms": comp_fwd, "comp_cv_pct": _cv(rp), "attn_cv_pct": _cv(ra),
        "pcie_comm_meas_fwd_ms": pcie_comm_meas,   # raw PCIe comm, DISCARDED (sanity only)
        # analytic comm volume (bus traffic, bytes)
        "msg_bytes": mB, "ar_bus_bytes": bus_v, "ag_gathered_bytes": ag_g,
        "ag_bus_bytes": ag_bus, "tot_bus_bytes": tot_bus,
        # PROJECTED A100-NVLink comm time (volume / busbw)
        "proj_ar_fwd_ms": proj_ar, "proj_ag_fwd_ms": proj_ag, "proj_comm_fwd_ms": proj_comm,
        "comm_frac_pct": comm_frac,                # projected comm / (proj comm + measured comp)
        "comm_pct_capture": comm_pct,
        # per output token
        "committed_per_step_mean": st.mean(com),
        "tw_comm_per_tok_ms": tw_comm, "tw_comp_per_tok_ms": tw_comp,
        "bus_bytes_per_tok": tw_vol, "coverage": cov,
        # per-block structure (counter)
        "intrinsic_sk_mean": st.mean(ski), "batch_sk_mean": st.mean(skb),
        "straggler_waste_pct": 100 * st.mean(waste) / st.mean(skb),
    }
    json.dump(metrics, open(f"{blocks_prefix}_a100metrics.json", "w"), indent=2)
    return metrics


def sweep(logs_dir):
    files = glob.glob(os.path.join(logs_dir, "*_a100metrics.json"))
    files.sort(key=lambda p: int(re.search(r"_c(\d+)_", os.path.basename(p)).group(1))
               if re.search(r"_c(\d+)_", os.path.basename(p)) else 0)
    ms = [json.load(open(f)) for f in files]
    if not ms:
        print("  (no *_a100metrics.json found)")
        return
    print("\n===== D2 / A100 batch-size sweep — PROJECTED-NVLink comm vs MEASURED A100 compute =====")
    print(f"  comm = analytic volume / {BUSBW_REFS[BUSBW_HEADLINE]/1e9:.0f} GB/s ({BUSBW_HEADLINE}); "
          f"compute = MEASURED A100 (per-replay median, real)")
    h = (f"{'tag':16s} {'bs':>3} {'volMB/fwd':>9} {'projComm':>9} {'measComp':>9} {'comm%':>6} "
         f"{'comm/tok':>9} {'comp/tok':>9} {'s_k':>5} {'waste%':>6} {'tok/fwd':>7}")
    print(h)
    for m in ms:
        print(f"{m['tag']:16s} {m['dominant_bs']:>3d} {m['tot_bus_bytes']/1e6:>9.1f} "
              f"{m['proj_comm_fwd_ms']:>8.3f}m {m['comp_fwd_ms']:>8.3f}m {m['comm_frac_pct']:>5.1f}% "
              f"{m['tw_comm_per_tok_ms']:>8.4f}m {m['tw_comp_per_tok_ms']:>8.3f}m "
              f"{m['intrinsic_sk_mean']:>5.1f} {m['straggler_waste_pct']:>5.0f}% "
              f"{m['committed_per_step_mean']:>7.2f}")
    print("\n  volMB/fwd = total ring bus traffic per forward (41 all-reduces + 1 vocab all-gather).")
    print("  projComm = PROJECTED A100-NVLink comm/fwd (volume/busbw); measComp = MEASURED A100 compute/fwd.")
    print("  comm% = projComm/(projComm+measComp) = projected comm fraction of the forward on NVLink.")
    print("  NOTE: volume/busbw is a bandwidth model; the 41 small all-reduces are partly latency-bound,")
    print("        so projComm (hence comm%) is a LOWER bound — see the H100 sibling's measured ~22-32%.")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        sweep(sys.argv[2])
    elif len(sys.argv) == 3:
        single(sys.argv[1], sys.argv[2])
    else:
        sys.exit("usage: parse_a100.py <blocks_prefix> <nsys_rep> | --sweep <logs_dir>")


if __name__ == "__main__":
    main()
