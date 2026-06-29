# FOCUS Phase-A Smoke Test (single A100, LLaDA2.0-mini)

Validates the FOCUS dLLM token-eviction algorithm end-to-end on one A100 80GB.

## What it checks
1. **FOCUS runs end-to-end** on a real model (LLaDA2.0-mini, TP=1).
2. **Correctness anchor:** FOCUS with `alpha -> inf` (budget `K` saturates to the
   full block, so *no* eviction) must produce **bit-for-bit identical** output to
   the `LowConfidence` baseline.
3. **Quality under eviction:** FOCUS with `alpha=1.5` (real eviction, ~13/32
   tokens retained per step) still produces correct, coherent generations.

## Run
```bash
cd experiments/dllm/focus_a100_smoke
./run_smoke.sh focus_alpha_inf   # anchor (no eviction)
./run_smoke.sh low_confidence    # baseline
./run_smoke.sh focus             # real eviction (alpha=1.5)
```
Each writes `logs/<variant>_gen.json`. Compare the anchor vs baseline:
```bash
python - <<'PY'
import json
lc=json.load(open('logs/low_confidence_gen.json'))
fi=json.load(open('logs/focus_alpha_inf_gen.json'))
print('MATCH' if all(a['text']==b['text'] for a,b in zip(lc,fi)) else 'DIFFER')
PY
```

## Result (2026-06-29)
- `focus_alpha_inf` vs `low_confidence`: **4/4 prompts MATCH** ✓
- `focus` (alpha=1.5): coherent + correct (43; primes 2,3,5,7,11; valid `add()`) ✓

## Notes
- `--mem-fraction-static 0.7` gives sparse-MoE OOM headroom on a single card
  (see memory `llada2-launch-config-a100`).
- `--disable-cuda-graph --attention-backend flashinfer` per the dLLM eager path.
- Phase A uses a **logit-masking** realization of eviction (full forward, then
  suppress commits on evicted positions): correct, but no FLOPs savings yet. The
  reduced (split) forward that realizes the speedup is Phase C.
