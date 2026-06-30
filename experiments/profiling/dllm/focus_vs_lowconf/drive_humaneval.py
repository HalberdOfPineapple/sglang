#!/usr/bin/env python3
"""Sustained-concurrency load driver for the FOCUS-vs-LowConfidence experiment.

Keeps exactly ``concurrency`` /generate requests in flight (cycling the first N
HumanEval prompts) so the running batch stays full, then reports throughput and
the per-request latency distribution. Adapted from the D2 driver; adds per-request
latency capture (dumped to RESULT_JSON) so the parser can compute mean/median/p90
latency and tok/s for each algorithm. stdlib only; bypasses the http proxy.

Usage:
  drive_humaneval.py HOST PORT CONCURRENCY TOTAL_REQUESTS PROMPTS_GZ N_SAMPLES MAX_NEW [RESULT_JSON]
"""
import gzip
import json
import os
import sys
import threading
import time
import urllib.request


def load_prompts(path: str, n: int):
    op = gzip.open if path.endswith(".gz") else open
    prompts = []
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    conc, total = int(sys.argv[3]), int(sys.argv[4])
    prompts_path, n_samples, max_new = sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
    result_json = sys.argv[8] if len(sys.argv) > 8 else None
    prompts = load_prompts(prompts_path, n_samples)
    url = f"http://{host}:{port}/generate"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy
    lock = threading.Lock()
    state = {"issued": 0, "done": 0, "out_tokens": 0}
    latencies = []  # per-request wall latency (s)
    t0 = time.time()

    def post(prompt: str):
        body = json.dumps(
            {
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        ts = time.time()
        try:
            r = opener.open(req, timeout=900)
            d = json.loads(r.read())
            n = d.get("meta_info", {}).get("completion_tokens", 0)
        except Exception as e:  # noqa: BLE001 — load gen, keep going
            sys.stderr.write(f"[drive] req error: {e}\n")
            n = 0
        lat = time.time() - ts
        with lock:
            state["done"] += 1
            state["out_tokens"] += n
            if n > 0:
                latencies.append(lat)

    def worker():
        while True:
            with lock:
                if state["issued"] >= total:
                    return
                idx = state["issued"]
                state["issued"] += 1
            post(prompts[idx % len(prompts)])

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    tok_s = state["out_tokens"] / dt if dt > 0 else 0.0
    lat_sorted = sorted(latencies)

    def pct(p):
        if not lat_sorted:
            return 0.0
        i = min(len(lat_sorted) - 1, int(p * len(lat_sorted)))
        return lat_sorted[i]

    summary = {
        "concurrency": conc,
        "total_reqs": total,
        "done": state["done"],
        "out_tokens": state["out_tokens"],
        "wall_s": dt,
        "tok_s": tok_s,
        "lat_mean_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "lat_p50_s": pct(0.50),
        "lat_p90_s": pct(0.90),
        "max_new": max_new,
    }
    print(
        f"[drive] done={summary['done']} reqs  out_tokens={summary['out_tokens']}  "
        f"{dt:.1f}s  {tok_s:.0f} tok/s  lat(mean/p50/p90)="
        f"{summary['lat_mean_s']:.1f}/{summary['lat_p50_s']:.1f}/{summary['lat_p90_s']:.1f}s "
        f"(conc={conc})"
    )
    if result_json:
        os.makedirs(os.path.dirname(result_json), exist_ok=True)
        with open(result_json, "w") as f:
            json.dump({"summary": summary, "latencies": latencies}, f, indent=2)
        print(f"[drive] wrote {result_json}")


if __name__ == "__main__":
    main()
