# EvoPlace: LLM-Guided Evolutionary Discovery of Placement Objective Functions

**[DRAFT — updated as experiments complete]**  
Target: ISPD 2026 / DAC 2026  

---

## Abstract

[To be written after Exp 5 results]

DREAMPlace established GPU-accelerated differentiable placement as the dominant paradigm, achieving 40× speedups over CPU placers. However, its objective function components — the WA-WL smoothness schedule γ(t), the density weight schedule λ(t), and the electrostatic density model — are hand-tuned heuristics that optimize half-perimeter wirelength (HPWL), a metric with poor correlation (ρ < 0.28) with post-route timing (TNS). We present EvoPlace, a framework that applies LLM-guided evolutionary search (OpenEvolve) to automatically discover better objective function components for differentiable placement. By evolving Python functions for γ and λ scheduling using TNS as fitness, EvoPlace discovers schedules that...

---

## 1. Introduction

VLSI placement determines the physical location of millions to billions of cells on a chip. Placement quality directly impacts wirelength, routability, timing closure, and power consumption. The dominant approach for the past decade has been analytical global placement: formulate cell positions as continuous variables and minimize a differentiable objective combining wirelength and density penalties via gradient descent [ePlace, RePlAce, DREAMPlace].

DREAMPlace [Lin+ DAC 2019] was a breakthrough: by implementing placement as a deep learning computation graph on GPU, it achieved 40× speedup over CPU-based analytical placers. The key insight was that the Nesterov optimizer, FFT-based density solver, and weighted-average (WA) wirelength model all map naturally to GPU tensor operations.

However, DREAMPlace and its successors have a fundamental limitation: **they optimize HPWL, not timing**. Recent analysis by ChiPBench [2024] shows that AI-based placers "perform poorly in end-to-end PPA metrics compared to OpenROAD, particularly in Power, TNS, and Area." The root cause is that HPWL and post-route TNS have rank correlation below 0.28 — a weak proxy at best.

Moreover, all the algorithmic components of DREAMPlace — the smoothness schedule γ(t) controlling WA-WL approximation quality, the density weight schedule λ(t) balancing spreading vs. wirelength, and the preconditioning strategy — are hand-tuned heuristics. AutoDMP [2023] used Bayesian optimization over DREAMPlace's hyperparameters but operated at the configuration level, not the algorithm level.

**Our contribution**: We apply LLM-guided evolutionary code search to discover placement algorithm components that are fundamentally better than human-designed heuristics, using TNS (not HPWL) as the primary fitness metric. Our framework, EvoPlace, treats each algorithmic component as a Python function to be evolved by a Claude LLM ensemble using MAP-Elites population diversity maintenance.

Key results [placeholder — to be filled after experiments]:
- Evolved γ schedule: X% HPWL improvement, Y% faster convergence (Exp 1)
- Evolved λ schedule: Z% reduction in divergence events (Exp 2)
- GNN warm initialization: W% fewer iterations (Exp 3)
- Timing surrogate loss: V% TNS improvement over DREAMPlace 4.0 (Exp 4)
- Full system: U% HPWL + T% TNS improvement on ISPD 2015 + ICCAD 2015 (Exp 5)

---

## 2. Background

### 2.1 Differentiable Global Placement

The global placement objective minimizes half-perimeter wirelength (HPWL) subject to a density constraint:

```
min_{x,y}  Σ_{e∈E} WL(e; x, y)
s.t.        D(x, y) ≤ D̂
```

Where WL(e; x, y) is the half-perimeter wirelength of net e, and D(x, y) is the cell density (overlap). This is converted to an unconstrained problem via augmented Lagrangian:

```
f = Σ_{e∈E} WL(e; v) + ⟨λ, D(v, r) + (1/2)μP_λ ⊙ D²(v, r)⟩
```

**WA-WL approximation** [Hsu+ TCAD 2013]: The true HPWL is non-differentiable; the weighted-average model approximates it:

```
WL_WA(e; γ) = [Σᵢ xᵢ·exp(xᵢ/γ) / Σᵢ exp(xᵢ/γ)] - [Σᵢ xᵢ·exp(-xᵢ/γ) / Σᵢ exp(-xᵢ/γ)]
```

As γ → 0, WL_WA → HPWL but gradients vanish. As γ → ∞, WL_WA becomes smooth but inaccurate. DREAMPlace uses a fixed schedule γ₀ → γ_min.

**Electrostatic density model** [ePlace]: Cell density is modeled by analogy with electrostatics. Cells are "charges," and the Poisson equation ∇²ψ = ρ (solved via FFT) gives a density force field.

**Nesterov's optimizer with preconditioning**: DREAMPlace uses Nesterov momentum with a diagonal preconditioner estimated from pin count (WL Hessian) and cell area (density Hessian).

### 2.2 DREAMPlace 3.0 Multi-Electrostatics

For fence-region constraints, DREAMPlace 3.0 maintains one electrostatic field per region, computed in parallel. Cells belonging to region k see only field k's force. Virtual blockages fill non-region areas to prevent cell drift. This gives O(Σ|Vk|) = O(|V|) complexity.

### 2.3 Known Limitations

[See NOTES.md for detailed analysis]

The density weight schedule λ(t) uses exponentially increasing step size:

```
λ ← min(λ_max, λ + α · ∇̂_λf / ‖∇̂_λf‖₂)
α ← γ(D, P_λ) · α
```

Both γ and the step-size multiplier are heuristic. Pathological cases: slow spreading (30–50% of runtime), optimizer divergence, saddle-point trapping.

### 2.4 OpenEvolve and LLM-Guided Code Evolution

OpenEvolve [algorithmicsuperintelligence, 2024] is an open-source implementation of Google DeepMind's AlphaEvolve, which uses LLM ensembles to evolve programs via MAP-Elites quality-diversity optimization. Key components:

- **MAP-Elites**: Maintains a population grid indexed by behavioral features (quality axes). Encourages exploration across the feature space.
- **Island model**: Multiple isolated populations evolve independently with periodic migration, preventing premature convergence.
- **Cascade evaluation**: Three-stage filtering rejects poor programs early before expensive full evaluation.
- **LLM mutation**: Claude generates modified versions of functions; the diff-based mutation mode makes targeted changes.

We adapt OpenEvolve to evolve Python functions representing placement algorithm components, with our VLSI placement harness as the evaluator.

### 2.5 Autoresearch Loop

Karpathy's autoresearch [2024] demonstrates fully autonomous ML experimentation: an AI agent edits a training script, runs experiments for a fixed time budget, reads results, and iterates — achieving ~100 experiments per GPU-night. We adapt this pattern for placement: fixed evaluation harness, editable algorithm components, TSV result logging.

---

## 3. EvoPlace Framework

[Architecture diagram placeholder]

### 3.1 Evaluation Harness (Fixed)

The evaluation harness wraps DREAMPlace's Python API and is never modified during evolution. Given a placement component (γ schedule, λ schedule, or initialization function), it:

1. Loads benchmark netlist (ISPD 2015 or ICCAD 2015 format)
2. Initializes DREAMPlace with the component injected via hook
3. Runs global placement for N iterations or until overflow < threshold
4. Returns metrics dict: `{hpwl, overflow, tns_proxy, runtime_s, divergence_events}`

**TNS proxy**: We use a static timing analysis approximation — sum of max-slack violations across all nets weighted by net criticality — as a differentiable proxy for true post-route TNS. (Full router TNS too expensive for evolution loop.)

**Cascade evaluation**: 
- Stage 1 (50 iterations, <30s): Reject if overflow doesn't decrease
- Stage 2 (200 iterations, <2min): Reject if HPWL > 1.5× baseline
- Stage 3 (full run, <10min): Record all metrics for surviving candidates

### 3.2 Evolution Engine

[To be written after Exp 1 confirms the framework works]

### 3.3 Autoresearch Loop

[To be written after Exp 1]

---

## 4. Experiment 1: WL Smoothing Schedule Evolution

[Results placeholder — to be filled after Exp 1]

**Setup**: Evolved function signature:
```python
def gamma_schedule(iteration: int, total_iterations: int, 
                   overflow: float, hpwl_history: list[float]) -> float:
    """Return γ smoothness parameter for WA-WL at this iteration."""
    ...
```

**Baseline**: DREAMPlace default — linear decay from γ₀=8.0 to γ_min=0.5 over total_iterations.

**Fitness**: Final HPWL on `ispd2015/fft_1` (small benchmark, fast to evaluate). Lower is better.

**Evolution config**: 4 islands, 200 iterations, cascade thresholds [2.0, 1.3].

**Results**: [PENDING]

---

## 5. Experiment 2: Density Weight Schedule Evolution

[Results placeholder — to be filled after Exp 2]

**Setup**: Evolved function signature:
```python
def lambda_schedule(iteration: int, overflow: float, 
                    overflow_history: list[float], 
                    gradient_norm: float,
                    current_lambda: float) -> float:
    """Return new density weight λ at this iteration."""
    ...
```

**Baseline**: DREAMPlace exponential increase based on density level.

**Fitness**: HPWL × (1 + 5 × divergence_events). Penalizes instability.

**Results**: [PENDING]

---

## 6. Experiment 3: GNN Warm Initialization

[Results placeholder — to be filled after Exp 3]

**Architecture**: Heterogeneous GNN on cell-net hypergraph.
- Cell nodes: features = [area, pin_count, cell_type_embedding]
- Net nodes: features = [fanout, net_weight]  
- Cell→Net edges: pin offset (x, y)
- 4 message-passing layers, hidden dim 128
- Output: (x_pred, y_pred) per cell, normalized to [0, 1]

**Training data**: DREAMPlace final placements on ISPD 2015 training split.

**Evaluation**: #iterations to reach overflow < 0.2 vs. center initialization.

**Results**: [PENDING]

---

## 7. Experiment 4: Differentiable Timing Surrogate

[Results placeholder — to be filled after Exp 4]

**Surrogate architecture**: 
- Input: per-net features (net center distance, fanout, critical-path flag, driver/sink cell sizes)
- MLP: 3 layers, 256 hidden, ReLU, BatchNorm
- Output: per-net timing slack approximation
- TNS proxy = Σ max(0, -slack_pred) across nets

**Integration**: Added as extra loss term in DREAMPlace's augmented Lagrangian:
```
f = Σ WL(e; v) + ⟨λ, D(v, r)⟩ + β · TNS_surrogate(v)
```

**Fitness**: True TNS proxy on ICCAD 2015 benchmarks (not HPWL).

**Results**: [PENDING]

---

## 8. Experiment 5: Full System

[Results placeholder — to be filled after Exp 5]

**Stack**: GNN init + evolved γ schedule + evolved λ schedule + timing surrogate loss.

**Comparison targets**:
- DREAMPlace 3.0 (ICCAD 2020 paper baseline)
- DREAMPlace 4.0 (timing-driven, TCAD 2023)
- OpenROAD RePlAce (production CPU-based)

**Target**: ≥5% HPWL + ≥10% TNS improvement vs. DREAMPlace 4.0.

**Results**: [PENDING]

---

## 9. Conclusion

[To be written after Exp 5]

---

## References

[Gu+ ICCAD 2020] Jiaqi Gu, Zixuan Jiang, Yibo Lin, David Pan. DREAMPlace 3.0: Multi-Electrostatics Based Robust VLSI Placement with Region Constraints.

[Lin+ DAC 2019] Yibo Lin et al. DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement.

[Liao+ TCAD 2023] DREAMPlace 4.0: Timing-Driven Placement with Momentum-Based Net Weighting.

[Agnesina+ ISPD 2023] AutoDMP: Automated DREAMPlace-Based Macro Placement.

[Lu+ ISPD 2023] DREAM-GAN: Advancing DREAMPlace towards Commercial-Quality.

[Lai+ ICLR 2023] ChiPFormer: Transferable Chip Placement via Offline Decision Transformer.

[Lai+ NeurIPS 2022] MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning.

[LAMPlace ICLR 2025] Learning to Optimize Cross-Stage Metrics in Macro Placement.

[MLBuf-RePlAce MLCAD 2025] Recursive Learning-Based Virtual Buffering for Analytical Global Placement.

[Cheng+ CACM 2023] Reevaluating Google's Reinforcement Learning for IC Macro Placement.

[ChiPBench 2024] Benchmarking End-To-End Performance of AI-Based Chip Placement Algorithms.

[OpenEvolve 2024] algorithmicsuperintelligence/openevolve: Open-source AlphaEvolve implementation.

[Karpathy autoresearch 2024] karpathy/autoresearch: Autonomous ML research framework.
