# EvoPlace Autoresearch Program

## Goal
Improve VLSI placement quality (minimize val_norm_hpwl) by modifying the scheduling functions in `train.py`.

## Current Best
val_norm_hpwl: 1.0000 (baseline, Exp 0 not yet run on hardware)

## What You're Optimizing
The `gamma_schedule` and `lambda_schedule` functions in `train.py` control:

1. **gamma_schedule**: The smoothness parameter γ for the WA-WL wirelength approximation
   - High γ early: helps pull clustered cells to their targets
   - Low γ late: accurate HPWL gradients for fine-tuning
   - Key insight: the schedule should adapt to the current optimization state

2. **lambda_schedule**: The density penalty weight λ
   - Starts low: allow cells to overlap while finding good wirelength
   - Grows: gradually spread cells apart
   - Challenge: divergence if λ grows too fast; slow convergence if too slow

## Evaluation
Run `python autoresearch/evaluate.py` to get val_norm_hpwl.
Results are logged to `autoresearch/results.tsv`.

## Constraints
- Only modify function bodies in train.py
- Keep exact signatures
- Only use built-in Python + math + numpy
- No file I/O, no network calls

## Research Directions

**Phase 1 (current)**: Find a better gamma_schedule
- Try: overflow-adaptive, gradient-norm-adaptive, piecewise linear, cosine annealing
- Hypothesis: adaptive schedules that react to the current state outperform fixed schedules

**Phase 2**: Find a better lambda_schedule
- Try: plateau-detector-based boost, gradient-magnitude-aware, entropy-injection-triggered

**Phase 3**: Joint optimization
- Try: schedules that coordinate gamma and lambda updates

## Key Reference
DREAMPlace 3.0 paper (ICCAD 2020, Gu et al.):
- Plateau detection: PLT = (max_L(OVFL) - min_L(OVFL)) / avg_L(OVFL) < δ_PLT when OVFL > 0.9
- When plateau detected: double density weight + entropy injection
- This is the heuristic we're trying to improve

## Logging Format
The evaluator prints:
```
val_norm_hpwl:     0.982000   ← primary metric (lower = better)
hpwl:              4.12e+08
mean_overflow:     0.0650
tns_proxy:         8.40e+06
runtime_seconds:   45.2
divergence_events: 1
```
