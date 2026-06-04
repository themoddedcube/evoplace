# EvoPlace: LLM-Guided Evolutionary VLSI Placement

**Status**: Active research | **Target venue**: ISPD 2026 / DAC 2026

EvoPlace is a research framework that uses LLM-guided evolutionary search (inspired by [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)) and autonomous experiment loops (inspired by [autoresearch](https://github.com/karpathy/autoresearch)) to discover better VLSI placement algorithm components than the hand-tuned ones in DREAMPlace.

## The Problem with DREAMPlace

DREAMPlace (DAC 2019, ICCAD 2020) achieved a 40× speedup by treating placement as differentiable GPU optimization. But it has three structural weaknesses:

1. **Wrong metric**: HPWL has rank correlation < 0.28 with post-route TNS (ChiPBench 2024). Optimizing HPWL doesn't close timing.
2. **Heuristic schedules**: The density weight (λ) and WL smoothness (γ) schedules are hand-tuned constants, not learned.
3. **Spreading bottleneck**: 30–50% of runtime is spent on cell spreading; divergence requires entropy injection as a workaround.

## Our Approach

We treat each placement algorithm component as a Python function to be evolved by an LLM ensemble:

```
DreamPlace backbone
    └── γ(t) = WL smoothing schedule      ← evolved by OpenEvolve (Exp 1)
    └── λ(t) = density weight schedule    ← evolved by OpenEvolve (Exp 2)
    └── init(netlist) = starting positions ← GNN predictor (Exp 3)
    └── L_timing = timing surrogate loss   ← differentiable MLP (Exp 4)
```

Evolution fitness uses **TNS proxy** as primary metric, not HPWL.

## Experiments

| # | Name | Technique | Expected Gain |
|---|------|-----------|---------------|
| 0 | Baseline | DREAMPlace 4.0 reproduction | — |
| 1 | WL Smoothing | Evolve γ schedule | 1–3% HPWL, 10–20% faster |
| 2 | Density Schedule | Evolve λ schedule | Eliminate 80% divergence events |
| 3 | GNN Init | Warm initialization | 30–50% fewer iterations |
| 4 | Timing Loss | Differentiable TNS surrogate | 10–15% TNS improvement |
| 5 | Full System | Best of Exp 1–4 | ≥5% HPWL + ≥10% TNS vs DREAMPlace 4.0 |

## Repository Structure

```
evoplace/
├── evaluator/          # FIXED evaluation harness (run_placement.py, metrics.py)
├── dreamplace_ext/     # DreamPlace hooks (custom_objectives.py, schedulers.py)
├── evolve/             # OpenEvolve adaptation (MAP-Elites over algorithm components)
├── autoresearch/       # Autonomous experiment loop (evaluate.py FIXED, train.py EDITABLE)
├── models/             # GNN initializer, timing surrogate MLP
├── experiments/        # Per-experiment configs, logs, and results
├── graphs/             # Auto-generated plots (convergence, HPWL bars, Pareto)
├── benchmarks/         # ISPD 2015, ICCAD 2015 circuits (not committed, see below)
├── vendor/dreamplace/  # DreamPlace fork (submodule)
├── NOTES.md            # Running research journal (updated after every experiment)
└── PAPER_DRAFT.md      # Paper draft sections
```

## Benchmarks

**ISPD 2015** (macro placement with region constraints):
```
wget http://www.ispd.cc/contests/15/ispd2015_contest.tar.gz
tar -xzf ispd2015_contest.tar.gz -C benchmarks/ispd2015/
```

**ICCAD 2015** (timing-driven placement, requires free registration):  
Register at http://iccad-contest.org/2015/ → download to `benchmarks/iccad2015/`

**NanGate 45nm PDK** (open source):
```
wget https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/raw/master/flow/platforms/nangate45/...
```

## Hardware Requirements

- Linux (Ubuntu 22.04+), CUDA 11.8+, Python 3.10+
- Exp 0–2: 1× GPU ≥16GB VRAM (RTX 3090, A100)
- Exp 3–5: 2–4× GPUs, 32GB RAM
- Total estimated: ~150–250 GPU-hours (~1 A100-week)

## Quick Start (once hardware is ready)

```bash
# 1. Build DreamPlace
cd vendor/dreamplace
pip install -r requirements.txt
python setup.py build_ext --inplace

# 2. Verify harness works
cd ../..
python evaluator/run_placement.py --benchmark benchmarks/ispd2015/fft_1 --output experiments/exp00_baseline/

# 3. Run baseline (Exp 0)
python experiments/exp00_baseline/run.py

# 4. Start evolution (Exp 1, requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python evolve/run_evolution.py --experiment exp01_wl_smoothing
```

## Key Files

| File | Status | Purpose |
|------|--------|---------|
| `evaluator/run_placement.py` | FIXED | Runs DreamPlace, returns metrics dict |
| `evaluator/metrics.py` | FIXED | HPWL, overflow, TNS proxy computation |
| `autoresearch/evaluate.py` | FIXED | Evaluation harness for autoresearch loop |
| `autoresearch/train.py` | EDITABLE | Current placement config under research |
| `autoresearch/results.tsv` | APPEND-ONLY | Experiment log |

## Paper

**Title**: "EvoPlace: LLM-Guided Evolutionary Discovery of Placement Objective Functions"  
**Authors**: [themoddedcube]  
**Target**: ISPD 2026 or DAC 2026  
Draft sections in `paper/sections/`; see `PAPER_DRAFT.md` for current state.

## Citation

```bibtex
@article{evoplace2026,
  title={EvoPlace: LLM-Guided Evolutionary Discovery of Placement Objective Functions},
  author={},
  booktitle={Proceedings of ISPD 2026},
  year={2026}
}
```
