# Running Experiments

Activate the venv first: `source ~/evoplace_venv/bin/activate`

## Sanity gates (run before ANY evolution campaign)

```bash
# 1. Seed program must reproduce the default baseline (expect norm_hpwl ≈ 1.0 ± 0.01)
python -c "
from evolve.evaluator_wrapper import evaluate
r = evaluate('evolve/initial_program.py')
print(r['metrics']['normalized_hpwl'])"

# 2. Differential liveness probe: a degenerate schedule (constant γ=0.01)
#    must score very differently (in practice: cascade-rejected).
#    If every candidate scores identically, the hooks are dead — STOP.
```

## Exp 0 — Baseline Reproduction

```bash
# Quick test (fft_1, fft_2 only)
python experiments/exp00_baseline/run.py --suite small

# Full ISPD 2015 suite (GPU recommended)
python experiments/exp00_baseline/run.py --suite ispd2015_no_region
```

Results are saved to `experiments/exp00_baseline/results/`. After re-baselining
on new hardware, update `BASELINE_HPWL` **and** `BASELINE_HPWL_STAGES` in
`evolve/evaluator_wrapper.py` (stage-matched 50/300-iteration baselines — see
SPARK_SETUP.md §6 for the measurement snippet).

## Exp 1 — γ Schedule Evolution

```bash
# Auto-detects LLM backend: Claude Code CLI → Anthropic API
python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 200

# Force a specific backend
python evolve/run_evolution.py --experiment exp01_wl_smoothing \
    --backend claude-code-cli --iterations 200

# Quick smoke test
python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 3
```

**LLM backends** (in auto-detection order):

| Backend | Requirement | Notes |
|---------|-------------|-------|
| `claude-code-cli` | Claude Code installed | `claude -p`; no key needed |
| `anthropic-api` | `ANTHROPIC_API_KEY` set | Direct API access |

## Exp 2 — λ Schedule Evolution

```bash
python evolve/run_evolution.py --experiment exp02_density_schedule --iterations 200
```

## Multi-seed confirmation (run after EVERY campaign)

Single-seed rankings near the noise floor are unreliable — confirm before
claiming anything (see the [paper](../paper/paper.pdf) for why):

```bash
python scripts/multiseed_rerank.py --experiment exp01_wl_smoothing \
    --top 5 --seeds 42 43 44 45 46 --benchmarks fft_1 fft_2
```

## Visualizations

```bash
# Side-by-side evolved-vs-default GIF + density/potential/field animations
python scripts/make_comparison_gif.py \
    --program experiments/exp01_wl_smoothing/evolution_runs/candidate_0117.py \
    --benchmark fft_1 --seed 42 --interval 25 --fields all

# Lambda audit-finding GIF (guard-branch ablation seed vs default) + fields
python scripts/make_comparison_gif.py \
    --program experiments/exp02_density_schedule/evolution_runs/best_program.py \
    --hook lambda --benchmark fft_2 --seed 42 --interval 25 --fields all \
    --out-dir graphs/comparisons/fft_2_lambda_s42

# Single-run convergence showcase (no --program)
# embed mode: surfaces strip baked INSIDE convergence.gif — one file, no
# load-time drift on github.com (the per-field table layout couldn't sync
# the 4 separate GIFs without JS).
python scripts/make_comparison_gif.py --benchmark superblue12 --interval 25 \
    --fields all --fields-render embed \
    --out-dir graphs/comparisons/superblue12_showcase
```

With `--fields`, the density/potential/field surfaces are written to
`fields.gif` — all panels side by side in one file, with the exact frame
sequence, fps, and loop period of the main GIF (`--fields-render strip`,
the default). The panels are therefore mutually pixel-locked; against the
main GIF, equal loop periods keep any browser load-start offset constant,
and both files carry iteration counters so alignment is verifiable.
`--fields-render embed` draws the surfaces inside the main GIF instead;
`separate` writes legacy free-running per-field GIFs. Frame rate defaults
to `--iters-per-sec 250 / --interval`, so GIFs captured at different
intervals still advance their iteration counters at the same wall-clock
rate.

## Stub Mode (no DREAMPlace build required)

All experiments support `--stub` for synthetic results, enabling full pipeline testing without hardware:

```bash
python experiments/exp00_baseline/run.py --suite small --stub
python evaluator/run_placement.py --benchmark benchmarks/fft_1 --stub
```

## Tests

```bash
# All tests (excludes model tests that require torch)
python -m pytest tests/ -q --ignore=tests/test_models.py

# Full suite (requires torch in venv)
python -m pytest tests/ -q
```

114 tests cover: metrics math, baseline schedulers, GNN/surrogate forward+backward pass, full stub pipeline.

## Paper

```bash
cd paper && make   # requires tectonic (https://tectonic-typesetting.github.io)
```
