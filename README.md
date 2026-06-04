# EvoPlace: LLM-Guided Evolutionary VLSI Placement

> **Status**: Active research &nbsp;|&nbsp; **Target venue**: ISPD 2026 / DAC 2026 &nbsp;|&nbsp; **Python**: 3.10+ &nbsp;|&nbsp; **Framework**: DREAMPlace 4.0

EvoPlace is a research system that applies LLM-guided evolutionary search to automatically discover better algorithmic components for differentiable VLSI placement, targeting post-route timing slack (TNS) rather than the wirelength proxy (HPWL) that current tools optimize.

---

## Motivation

DREAMPlace (DAC 2019, ICCAD 2020) achieved a landmark 40× speedup by casting VLSI placement as GPU-accelerated differentiable optimization. Its successors remain the state of the art. But three structural weaknesses limit their usefulness for modern timing-closure flows:

| Weakness | Evidence |
|---|---|
| **Wrong metric** | HPWL has rank correlation ρ < 0.28 with post-route TNS (ChiPBench 2024) |
| **Heuristic schedules** | γ (WA-WL smoothness) and λ (density weight) are hand-tuned constants |
| **No timing awareness** | DREAMPlace 4.0 adds net weighting but still minimizes HPWL end-to-end |

ChiPBench (2024) confirms that AI-based placers "perform poorly in end-to-end PPA metrics compared to OpenROAD, particularly in TNS." EvoPlace directly addresses this gap.

---

## Approach

We treat each placement algorithm component as a Python function and evolve it using an LLM ensemble guided by TNS fitness:

```
DREAMPlace 4.0 backbone
├── γ(t, overflow, history)  ← Exp 1: evolved by OpenEvolve / Claude Code CLI
├── λ(t, overflow, history)  ← Exp 2: evolved by OpenEvolve / Claude Code CLI
├── init(netlist) → (x, y)   ← Exp 3: GNN warm initializer
└── L_timing(placement)      ← Exp 4: differentiable TNS surrogate
```

**Why evolutionary code search over hyperparameter tuning?** Bayesian optimization (AutoDMP, ISPD 2023) tunes scalars in a fixed functional form. We search over the *functional form itself* — the schedule shape, adaptive logic, and interaction with overflow — which is a strictly larger space.

**Why Claude Code CLI over a raw API key?** `claude -p` is available in any authenticated Claude Code session. No separate key management, same model quality, zero additional cost.

---

## Experiments

| # | Name | Method | Primary Metric | Status |
|---|------|--------|----------------|--------|
| 0 | DREAMPlace 4.0 Baseline | Reproduction | HPWL, TNS | ✅ **Done** — fft_1: 2.182e6 HPWL |
| 1 | WL Smoothing Schedule | Evolve γ(t) | HPWL ↓, convergence ↓ | 🔄 **Running** |
| 2 | Density Weight Schedule | Evolve λ(t) | Divergence events ↓ | ⏳ Pending |
| 3 | GNN Warm Initialization | Heterogeneous GNN | Iterations to converge ↓ | ⏳ Pending |
| 4 | Differentiable TNS Surrogate | MLP loss term | TNS ↓ | ⏳ Pending |
| 5 | Full System | Best of Exp 1–4 | ≥5% HPWL + ≥10% TNS | ⏳ Pending |

### Exp 0 Baselines (CPU, ISPD 2015)

Measured on WSL2 Ubuntu 24.04, Intel CPU, no GPU. GPU convergence (overflow ≤ 0.07) will produce lower HPWL values; these numbers set the CPU comparison floor.

| Circuit | HPWL | Overflow | Runtime |
|---------|------|----------|---------|
| `fft_1` | 2.182 × 10⁶ | 1.00 | 826 s |
| `fft_2` | 2.490 × 10⁶ | 1.00 | 78 s |

---

## Repository Structure

```
evoplace/
├── evaluator/              # Stable evaluation harness (never modified during experiments)
│   ├── run_placement.py    # DREAMPlace runner — cascade evaluation, stub fallback
│   ├── metrics.py          # HPWL, overflow, TNS proxy computation
│   └── benchmark_suite.py  # ISPD 2015 / ICCAD 2015 / small suite definitions
│
├── dreamplace_ext/         # DREAMPlace hook injection layer
│   ├── hooks.py            # γ, λ, init_positions, timing_loss hook singletons
│   ├── schedulers.py       # Baseline schedulers (linear, exponential, overflow-adaptive)
│   └── custom_objectives.py
│
├── evolve/                 # LLM-guided evolutionary search
│   ├── run_evolution.py    # CLI entry point; multi-backend LLM (CC CLI / API / Gemini)
│   ├── evaluator_wrapper.py # Bridges OpenEvolve ↔ DREAMPlace cascade evaluator
│   ├── initial_program.py  # Seed γ schedule (linear decay; evolution baseline)
│   └── config.yaml         # MAP-Elites settings, cascade thresholds
│
├── autoresearch/           # Karpathy-style autonomous experiment loop
│   ├── evaluate.py         # FIXED — stable evaluation entry point
│   └── train.py            # EDITABLE — current experiment under research
│
├── models/                 # PyTorch neural components (Exp 3, 4)
│   ├── gnn_placer.py       # Heterogeneous GNN for warm initialization
│   └── timing_surrogate.py # MLP differentiable TNS proxy
│
├── experiments/            # Per-experiment configs, results, logs
│   ├── exp00_baseline/
│   ├── exp01_wl_smoothing/
│   └── ...
│
├── graphs/                 # Auto-generated figures (convergence, HPWL bars, Pareto)
├── benchmarks/             # ISPD 2015 / ICCAD 2015 circuits (not committed — see below)
├── vendor/dreamplace/      # DREAMPlace 4.0 fork (git submodule)
├── scripts/setup_wsl.sh    # One-command WSL2 build + benchmark download
├── NOTES.md                # Running research journal
└── PAPER_DRAFT.md          # In-progress paper sections
```

---

## Setup

### Requirements

- Linux or WSL2 Ubuntu 22.04+ (tested on Ubuntu 24.04 in WSL2)
- Python 3.10+
- CMake ≥ 3.14, GCC ≥ 11, Boost ≥ 1.55
- PyTorch ≥ 2.0 (CPU sufficient for development; GPU required for convergent runs)
- CUDA 11.8+ for GPU placement (optional until DGX/GPU available)
- [Claude Code CLI](https://claude.ai/code) for LLM-guided evolution (no API key needed)

### Automated Setup (WSL2)

The `scripts/setup_wsl.sh` script handles all phases. Each phase can be run independently:

```bash
# Phase 1: system packages (requires sudo)
sudo bash scripts/setup_wsl.sh apt

# Phase 2: Python venv + PyTorch at ~/evoplace_venv
bash scripts/setup_wsl.sh python

# Phase 3: build DREAMPlace (~10–15 min)
bash scripts/setup_wsl.sh build

# Phase 4: download ISPD 2015 benchmarks + generate JSON configs
bash scripts/setup_wsl.sh benchmarks

# Phase 5: smoke test
bash scripts/setup_wsl.sh test
```

Or all at once: `sudo bash scripts/setup_wsl.sh all`

> **Note on paths with spaces**: DREAMPlace's C++ parser splits file paths on whitespace. The setup script creates a `~/evoplace` symlink to avoid this if your project directory contains spaces.

### Manual Build (any Linux)

```bash
# 1. Install system deps
sudo apt-get install -y cmake libboost-all-dev zlib1g-dev libomp-dev bison flex

# 2. Init DREAMPlace submodules
cd vendor/dreamplace
git submodule update --init thirdparty/pybind11 thirdparty/Limbo \
    thirdparty/OpenTimer thirdparty/munkres-cpp thirdparty/cub

# 3. Build (detect PyTorch ABI automatically)
CXX_ABI=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
mkdir -p ~/dreamplace_build && cd ~/dreamplace_build
cmake /path/to/evoplace/vendor/dreamplace \
    -DCMAKE_INSTALL_PREFIX=/path/to/evoplace/vendor/dreamplace/install \
    -DCMAKE_CXX_ABI=$CXX_ABI \
    -DPython_EXECUTABLE=$(which python3)
make -j$(nproc) && make install

# 4. Download benchmarks
cd /path/to/evoplace/vendor/dreamplace
python benchmarks/ispd2005_2015.py
```

---

## Running Experiments

Activate the venv first: `source ~/evoplace_venv/bin/activate`

### Exp 0 — Baseline Reproduction

```bash
# Quick test (fft_1, fft_2 only — ~15 min on CPU)
python experiments/exp00_baseline/run.py --suite small

# Full ISPD 2015 suite (GPU recommended — ~4–8 GPU-hours)
python experiments/exp00_baseline/run.py --suite ispd2015_no_region
```

Results are saved to `experiments/exp00_baseline/results/`. The script automatically updates `BASELINE_HPWL` in `evolve/evaluator_wrapper.py`.

### Exp 1 — γ Schedule Evolution

```bash
# Auto-detects LLM backend: Claude Code CLI → Anthropic API → Gemini
python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 200

# Force a specific backend
python evolve/run_evolution.py --experiment exp01_wl_smoothing \
    --backend claude-code-cli --iterations 200

# Quick smoke test (3 iterations, ~10 min on CPU)
python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 3
```

**LLM backends** (in auto-detection order):

| Backend | Requirement | Notes |
|---------|-------------|-------|
| `claude-code-cli` | Claude Code installed | Uses existing CC session; no key needed |
| `anthropic-api` | `ANTHROPIC_API_KEY` set | Direct API access |
| `gemini` | `GEMINI_API_KEY` set | Free tier at aistudio.google.com |

### Exp 2 — λ Schedule Evolution

```bash
python evolve/run_evolution.py --experiment exp02_density_schedule --iterations 200
```

### Stub Mode (no DREAMPlace build required)

All experiments support `--stub` for synthetic results, enabling full pipeline testing without hardware:

```bash
python experiments/exp00_baseline/run.py --suite small --stub
python evaluator/run_placement.py --benchmark benchmarks/fft_1 --stub
```

---

## Architecture Notes

### Evaluation Harness

`evaluator/run_placement.py` is the **fixed** evaluation contract — it is never modified during experiments. All algorithm components are injected as callables via `dreamplace_ext/hooks.py`:

```python
from evaluator.run_placement import run_placement

result = run_placement(
    benchmark_dir=Path("benchmarks/fft_1"),
    output_dir=Path("experiments/exp01/run_001"),
    gamma_schedule_fn=my_gamma_fn,   # inject evolved schedule
    max_iterations=2000,
)
print(result.metrics)  # {"hpwl": ..., "mean_overflow": ..., "tns_proxy": ...}
```

### Cascade Evaluation

To avoid spending full placement time on bad candidates, evolution uses three-stage cascade filtering (GPU mode):

```
Stage 1 (50 iters)  → reject if early HPWL trajectory is diverging
Stage 2 (300 iters) → reject if norm_hpwl > 1.3 × baseline
Stage 3 (full)      → complete placement; record all metrics
```

On CPU (no CUDA), cascade is replaced with a flat 50-iteration evaluation to avoid false elimination from non-convergent placements.

### Evolved Function Contract

The γ schedule function signature is fixed and must be preserved:

```python
def gamma_schedule(
    iteration: int,          # current step (0 to total_iterations-1)
    total_iterations: int,   # total planned steps
    overflow: float,         # current density overflow ∈ [0, 1]
    hpwl_history: list,      # HPWL values at previous iterations
) -> float:                  # γ ∈ [0.01, 20.0]
```

The evolution engine mutates only the function body. No new imports, no I/O, no external state.

---

## Key Design Decisions

**TNS as primary fitness, not HPWL.** HPWL and post-route TNS have rank correlation ρ < 0.28 (ChiPBench 2024). A placer that minimizes HPWL does not reliably close timing. All evolution fitness functions penalize TNS proxy; HPWL is tracked as a secondary metric.

**No RL.** Google's AlphaChip (Nature 2021) used RL macro placement; an independent evaluation (CACM 2023, arXiv:2306.09633) showed simulated annealing outperforms it in 17/17 cases. EvoPlace uses analytical global placement as its backbone.

**Algorithm evolution, not hyperparameter tuning.** AutoDMP (NVIDIA, ISPD 2023) applied Bayesian optimization over DREAMPlace's scalar hyperparameters. EvoPlace searches over the functional form of schedule components — a strictly larger and more expressive space.

**Stability boundary between fixed and editable code.** `evaluator/run_placement.py` and `autoresearch/evaluate.py` are never modified during experiments. This prevents fitness function drift and enables reproducible comparisons across evolution runs.

---

## Benchmarks

Benchmark files are not committed (too large). The setup script downloads them automatically; for manual download:

```bash
# ISPD 2015 (no registration required) — downloads ~170 MB
cd vendor/dreamplace && python benchmarks/ispd2005_2015.py

# ICCAD 2015 (timing-driven, free academic registration)
# Register at http://iccad-contest.org/2015/ → download to benchmarks/iccad2015/
```

**Circuits used:**

| Suite | Circuits | Primary use |
|-------|----------|-------------|
| `small` | `fft_1`, `fft_2` | Fast iteration during evolution |
| `ispd2015_no_region` | 11 circuits (fft, matrix_mult, superblue, ...) | Exp 0–2 baselines |
| `iccad2015` | 6 circuits with timing constraints | Exp 4–5 timing evaluation |

---

## Tests

```bash
source ~/evoplace_venv/bin/activate
cd evoplace

# All tests (excludes model tests that require torch)
python -m pytest tests/ -q --ignore=tests/test_models.py

# Full suite (requires torch in venv)
python -m pytest tests/ -q
```

114 tests cover: metrics math, baseline schedulers, GNN/surrogate forward+backward pass, full stub pipeline.

---

## Hardware

| Workload | Minimum | Recommended |
|----------|---------|-------------|
| Development + stub mode | Any CPU | Any CPU |
| CPU placement (non-convergent) | 8-core CPU | — |
| Exp 0–2 (convergent runs) | 1× GPU ≥ 16 GB | A100 / H100 |
| Exp 3–5 (GNN training + full system) | 1× GPU ≥ 24 GB | DGX H100 |

Estimated total compute: ~150–250 GPU-hours (≈ 1 A100-week).

---

## Paper

**Title**: EvoPlace: LLM-Guided Evolutionary Discovery of Placement Objective Functions  
**Target**: ISPD 2026 or DAC 2026  
**Draft**: `PAPER_DRAFT.md` / `paper/sections/`

---

## Citation

```bibtex
@inproceedings{evoplace2026,
  title     = {{EvoPlace}: {LLM}-Guided Evolutionary Discovery of Placement Objective Functions},
  author    = {},
  booktitle = {Proceedings of the ACM/IEEE International Symposium on Physical Design (ISPD)},
  year      = {2026}
}
```

---

## Related Work

| Paper | Venue | Relevance |
|-------|-------|-----------|
| DREAMPlace 4.0 — Liao et al. | DATE 2022 / TCAD 2023 | Backbone placer |
| AutoDMP — Agnesina et al. | ISPD 2023 | Bayesian hyperparameter tuning over DREAMPlace |
| ChiPBench — He et al. | arXiv 2024 | HPWL/TNS correlation analysis motivating this work |
| LAMPlace — Liao et al. | ICLR 2025 | Cross-stage metric learning for macro placement |
| OpenEvolve — Fati et al. | arXiv 2025 | LLM-guided evolutionary code search (our evolution engine) |
| AlphaChip rebuttal — Ma et al. | CACM 2023 | SA outperforms RL placement in 17/17 cases |
