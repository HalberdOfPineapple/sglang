"""Send a few fixed prompts to a running SGLang dLLM server and dump generations.

Deterministic (temperature 0 / greedy) so two algorithms can be diffed exactly.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Question: What is 17 + 26? Answer with just the number.",
    "Write a one-sentence definition of a diffusion language model.",
    "Complete the function:\n\ndef add(a, b):\n    return",
    "List the first five prime numbers separated by commas.",
]


def generate(host, port, prompt, max_new_tokens=128):
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    # Bypass any proxy for localhost.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--out", default="gen.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    results = []
    for i, prompt in enumerate(PROMPTS):
        try:
            out = generate(args.host, args.port, prompt, args.max_new_tokens)
            text = out.get("text", "")
            results.append({"idx": i, "prompt": prompt, "text": text})
            print(f"\n=== PROMPT {i} ===\n{prompt}\n--- GEN ---\n{text}")
        except Exception as e:  # noqa: BLE001
            print(f"[driver] prompt {i} FAILED: {e}", file=sys.stderr)
            results.append({"idx": i, "prompt": prompt, "error": str(e)})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if "text" in r and r["text"])
    print(f"\n[driver] {n_ok}/{len(PROMPTS)} prompts produced non-empty output")
    return 0 if n_ok == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
