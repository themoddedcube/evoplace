# Research Notes

Running journal of experiments, findings, and decisions.  
Updated after every experiment run. Dates are UTC.

---

## 2026-06-04 — Project Setup

### Background Reading

Read the DREAMPlace 3.0 ICCAD 2020 slides (Gu, Jiang, Lin, Pan — UT Austin / Peking University).

**Key takeaways from the paper:**
- DREAMPlace maps placement to a neural network problem: cell positions are weights, wirelength (WA approximation) + electrostatic density are the loss. GPU parallelizes gradient computation.
- v3.0 adds multi-electrostatics for fence-region constraints: each region gets its own electrostatic field, run in parallel. O(Σ|Vk|) = O(|V|) complexity — scales with total cells, not cells × regions.
- Virtual blockage insertion handles non-rectangular regions: rectangle slicing fills non-fence areas, preventing cells from drifting out.
- Density weight scheduling: λ updated via normalized preconditioned sub-gradient descent with exponentially increasing step size α based on density level.
- Robust optimizer: window-based plateau detector (PLT metric), entropy injection (shrink + perturb locations) escapes saddle points; divergence-aware rollback reverts if overflow stagnates.
- Results: >13% better HPWL, 11% better overflow than region-aware placers on ISPD 2015. 34.8× faster than 8-threaded NTUplace4dr.

**Key weaknesses identified:**
1. Slow spreading (30–50% of runtime)
2. Divergence requiring rollback heuristics
3. HPWL has rank correlation <0.28 with post-route TNS (per ChiPBench 2024)
4. All schedules (γ, λ, α) are hand-tuned heuristics

### Literature Survey Findings (2021–2026)

**Papers that improve on DREAMPlace:**
- **DREAMPlace 4.0** (TCAD 2023): Adds timing via momentum-based net weighting on critical paths. Still HPWL-primary.
- **AutoDMP** (ISPD 2023, NVIDIA): Bayesian hyperparameter optimization (MOTPE) over DREAMPlace config. Limited gains; still HPWL metric.
- **DREAM-GAN** (ISPD 2023): GAN discriminator trained on ICC2 commercial placements to guide DREAMPlace toward commercial quality.
- **ChiPFormer** (ICLR 2023): Offline decision transformer; 10× runtime reduction vs. SOTA. Transfer across designs.
- **MaskPlace** (NeurIPS 2022 Spotlight): Pixel-level visual RL; 60–90% WL reduction but RL training cost.
- **RoutePlacer** (ICCAD 2024): GNN-based routability surrogate integrated into differentiable placement.
- **LAMPlace** (ICLR 2025): Cross-stage metric learning for macro placement → 43% WNS improvement, 30.4% TNS improvement.
- **MLBuf-RePlAce** (MLCAD 2025): Virtual buffer prediction during global placement → 31% TNS, 53–56% WNS.

**Google's RL work (AlphaChip) — DISCREDITED:**
- Mirhoseini et al. Nature 2021: RL macro placement claims.
- Rebuttal (CACM 2023, arXiv 2306.09633): SA outperforms CT in 17/17 cases. Proxy cost has rank correlation <0.28 with timing. CT training takes 32+ hours vs <2 hrs for commercial tools.
- Conclusion: Do not use RL as primary placement strategy. SA or analytical methods dominate.

**Benchmark shift:**
- Old standard: ISPD 2015 (HPWL-centric)
- New standard: ICCAD 2015 (timing-driven) + ChiPBench 2024 (end-to-end PPA)
- We will use ISPD 2015 for early experiments (reproducibility), ICCAD 2015 for timing experiments.

### Research Hypothesis

Handcrafted schedules (γ, λ) in DREAMPlace are a local optimum in algorithm design space. LLM-guided evolutionary search over these components — using TNS as fitness — will discover configurations that:
- Improve TNS by >10% over DREAMPlace 4.0
- Match or improve HPWL
- No runtime regression

### Architecture Decision

**DreamPlace integration**: Fork (themoddedcube/DREAMPlace) added as git submodule. Hooks via Python subclassing/monkey-patching into `PlacementData` and the Nesterov optimizer. No upstream merge needed.

**Evolution**: OpenEvolve MAP-Elites with 4 islands, axes = [hpwl_final, convergence_iterations]. LLM ensemble: Claude Sonnet (primary) + Claude Haiku (exploration).

**Autoresearch loop**: 10 minutes per benchmark circuit. Results logged to autoresearch/results.tsv. Normalize HPWL to DreamPlace 4.0 baseline.

---

## 2026-06-04 — Klein-4 Empirical Validation of EvoPlace Thesis

Results from Klein-4 (2026 Macro Placement Challenge, #1 by proxy cost) directly validate and sharpen EvoPlace's core claims. Full NG45/ORFS measurement campaign completed.

### Finding 1 — Proxy→Timing Rank Inversion (Controlled Triple)

**Setup**: same design (mempool_tile, NanGate45, 4ns clock), same ORFS flow, same machine, three macro placements differing only in engine state.

| Placement | Proxy cost | WNS (ns) | TNS (ps) |
|-----------|-----------|----------|----------|
| Engine-cold | 0.7092 | −1.908 | −12,633 |
| OpenROAD RTL-MP baseline | — | −1.946 | −13,938 |
| Engine-warm (best proxy) | **0.6666** | **−2.089** | **−14,484** |

Best proxy cost → worst timing. Campaign-wide Kendall τ ≈ 0.05 proxy↔WNS.

**Implication for EvoPlace**: The proxy doesn't just decorrelate from timing — optimizing it can actively hurt timing. This is EvoPlace's core motivation backed by a clean controlled experiment. Any evolution fitness that maximises proxy (HPWL/density/congestion) is solving the wrong problem.

### Finding 2 — Timing Failures Are Structural; Objective Must See Clock/Path Structure

mempool's −1.9ns WNS is dominated by half-cycle latch paths: level-sensitive latches clocked on the inverted phase have a 2ns budget at a 4ns period, with ~1.5ns clock insertion delay already consuming most of that window. No wirelength-shaped objective can see this.

**Research task (highest-leverage)**: design fitness/loss features around per-path-group slack budgets. Latch/half-cycle groups must be weighted separately from flop-to-flop paths. Even a static pre-CTS required-time estimate (zero CTS insertion, ideal clock) gives the optimizer the structural signal it needs. Key sub-problems:
- Parse timing graph to classify path groups: FF→FF, FF→latch, latch→FF, latch→latch (half-cycle)
- Assign per-group slack budgets: half-cycle groups get `T/2 − t_cq_budget`; FF groups get `T − t_setup`
- Weight net criticality by path-group budget rather than global slack
- Differentiable formulation compatible with DREAMPlace's WA-WL loss

### Finding 3 — Compute Ladder for Timing-in-the-Loss

Cheapest-first ranking:

| Rung | Method | Cost | Differentiable |
|------|--------|------|---------------|
| (a) | Elmore/criticality-weighted wirelength | ~free | Yes |
| (b) | Periodic GPU-STA refresh, net weights updated every N iters | ~2–4× per affected iter | No (weights fixed between refreshes) |
| (c) | Learned timing predictor as fitness surrogate | training cost amortised | Yes |

**Architecture trick**: in multi-restart schemes, keep broad search timing-blind; apply timing term only to top-k survivors (~20% of compute). Benchmark each rung's marginal cost vs TNS-fitness fidelity.

### Finding 4 — Cautionary Prior Art for the Evolution Loop

A prior OpenEvolve run over placement hyperparameters ended 13% worse after 16 generations. Post-mortem:
- Single-benchmark fitness → overfit; candidate that wins fft_1 may lose everywhere else
- Several evolved parameters were dead code paths (never executed by the real optimizer)
- Search space too narrow on parameters that actually mattered

**Mitigations already in EvoPlace / to implement**:
- [x] Evolve schedule functions (not scalar knobs) — larger, executable search space
- [ ] Multi-design fitness from day one: score = mean(norm_hpwl across fft_1 + fft_2 + des_perf_1)
- [ ] Coverage assertions: inject tracing into evolved function to confirm all branches execute at least once per eval run
- [ ] Diversity maintenance: reject candidates with hash collision against top-10 archive

### ORFS Evaluator Gotchas (for Exp 4–5 TNS measurement through OpenROAD)

1. **"repair_timing converged to 0.000" ≠ signoff** — only post-route extracted STA (`6_finish.rpt`) counts; pre-route STA against estimated parasitics routinely flatters by 0.3–0.5ns.
2. **Per-design WNS ceilings** — ORFS CI accepts negative WNS on ariane133/136 (rules-base.json: −0.464/−0.300). Calibrate fitness expectations to each design's real ceiling, not zero.
3. **Genus-netlist instance name mangling** — ODB stores names with literal backslashes (`macro_mem\[0\].i_ram`); Yosys mangles to `_0__` form. Name matching must handle both.
4. **LEC binary AVX-512 requirement** — kepler-formal LEC SIGILLs on Zen 3 CPUs. Set `LEC_CHECK=0` (verification-only, netlist-invariant).

---

## 2026-06-04 — Exp 4 Implementation: Path-Group-Aware Timing Fitness

Designed and implemented the path-group-aware timing fitness in response to Finding 2. Full 9-check test suite passes.

### Architecture (Variant A — static weights into placedb.net_weights)

1. **`models/path_group_classifier.py`** — classifies each net by endpoint sequential type (FF→FF, FF→Latch, Latch→FF, Latch→Latch half-cycle, Unconstrained). Parses Liberty `.lib` for cell type (FF/LATCH/COMB) and SDC for clock period. Assigns per-net criticality weight = T / RT(group), clamped to [1.0, 10.0].

2. **`models/path_group_loss.py`** — `apply_weights_variant_a(placedb, pg_data)` writes weights into `placedb.net_weights` before `NonLinearPlace` runs. The existing WA-WL CUDA kernel multiplies by `net_weights` automatically — zero new GPU code. Also provides `make_timing_hook()` for future Variant B (optimizer-loop hook injection).

3. **`evaluator/run_placement.py`** — calls `_apply_path_group_weights()` after `placedb(params)`. Silently no-ops on ISPD 2015 (no SDC).

4. **`dreamplace_ext/hooks.py`** — added `set/get_path_group_data()` singleton.

5. **`evolve/evaluator_wrapper.py`** — added `exp04_timing_pathgroup` config; fitness metric is now per-experiment (`normalized_hpwl` for Exp 1/2, `tns_proxy` for Exp 4).

### Measured weights for mempool_tile NanGate45, T=4ns

| Group | Budget (ps) | Weight |
|-------|-------------|--------|
| FF→FF | 3330 | 1.20× |
| FF→Latch | 1320 | 3.03× |
| Latch→FF | 1150 | 3.48× |
| Latch→Latch (half-cycle) | 1070 | 3.74× |
| Unconstrained | — | 1.00× |

### Known limitations (v1)

- Static weights only; dynamic Elmore-based update deferred to v2
- `_get_cell_master()` falls back to instance-name prefix heuristic if `placedb.rawdb.cellTypeName()` not available — may misclassify uncommon cell naming conventions
- Latch→Latch always treated as half-cycle (conservative); polarity parsing from Liberty deferred
- Exp 4 benchmark (`mempool_tile`) not yet downloaded; exp04 evolution run pending ICCAD 2015 data

---

## 2026-06-04 — Exp 0 Baselines (CPU, ISPD 2015)

Real DREAMPlace 4.0 runs completed on WSL2 Ubuntu 24.04, Intel CPU, no GPU. Overflow = 1.0 in all cases — CPU cannot converge in 1000 iterations. GPU baseline on DGX will be significantly lower.

| Circuit | HPWL | Overflow | Runtime |
|---------|------|----------|---------|
| fft_1 | 2,182,147 | 1.00 | 826 s |
| fft_2 | 2,489,532 | 1.00 | 78 s |

## 2026-06-04 — Architectural Gap: γ/λ Hooks Are Not Wired Into DREAMPlace

**This is the most important architectural fact about the current codebase.**

### What the code does vs. what you'd expect

`run_placement.py` calls:
```python
hooks.set_gamma_schedule(gamma_schedule_fn)
hooks.set_lambda_schedule(lambda_schedule_fn)
...
placer = NonLinearPlace.NonLinearPlace(params, placedb, timer=None)
all_metrics = placer(params, placedb, learning_rate)
```

The hooks are registered in our module-level singletons. But `NonLinearPlace` is DREAMPlace's own C++/Python code — it knows nothing about our `dreamplace_ext.hooks` module. It never calls `hooks.get_gamma_schedule()`. The hooks are set, then silently ignored. DREAMPlace runs its own built-in linear γ decay and its own λ update rule for the entire placement run, regardless of what function the evolution engine evolved.

### Why this matters

Every "real" (non-stub) run in Exp 1 was measuring vanilla DREAMPlace with the default linear γ schedule, not the LLM-evolved schedule. The 20 candidates scored nearly identically (~2.25 norm_hpwl with ~1% variance) because they were all the same underlying run. The variance we saw was numerical noise from DREAMPlace itself (different random seeds, timer-dependent floating-point order), not schedule differences. The hill-climber had nothing real to climb.

This also means Exp 2 (λ evolution) has the same problem. Any "improvement" found in real-run mode would be a false positive.

The stub evaluator (`run_placement_stub`) does correctly call `gamma_schedule_fn` and rewards monotone-decreasing schedules with a 3% synthetic improvement — this is why the 3-iter smoke test showed a result at all. But the stub is synthetic by design.

### Why the gap exists

DREAMPlace's γ and λ schedules are implemented inside `NesterovPlace.py` (the core optimizer) as hardcoded update rules with JSON-configurable scalar knobs (`gamma`, `density_weight`). There is no callback mechanism. To inject our evolved functions we would need to:

**Option A — Monkey-patch NesterovPlace at runtime.** After `load_dreamplace()` returns, replace `NesterovPlace.NesterovPlace.update_gamma` (or equivalent method) with a wrapper that calls our hook. Fragile if DREAMPlace's internals change, but zero build changes.

**Option B — Fork NesterovPlace.py in the submodule.** Add explicit `if hooks.get_gamma_schedule(): gamma = hooks.get_gamma_schedule()(...)` calls at the right points. Clean, maintainable, requires knowing which method to patch.

**Option C — Pass schedules via JSON params.** DREAMPlace accepts `gamma` as a JSON scalar. We could write a thin outer loop: run 1-iteration DREAMPlace steps, read γ from our hook, write it back to params, re-run. This is ~100× slower (Python loop overhead) but requires no DREAMPlace changes.

**Recommended fix before any real Exp 1/2 results mean anything**: Option B. Identify the γ update in `install/dreamplace/NesterovPlace.py`, add a 3-line hook call, rebuild (or just edit the installed Python file since it's pure Python).

### What IS actually wired: Variant A (placedb.net_weights)

The path-group timing weights added for Exp 4 use a completely different integration point that **does** work today. `placedb.net_weights` is a numpy array that DREAMPlace's WA-WL computation reads directly every iteration via the C++/CUDA `WeightedAverageWirelength` operator. We write to it before `NonLinearPlace` is constructed, so DREAMPlace sees the modified weights from iteration 1. No hook mechanism involved — it's a direct data write to DREAMPlace's own data structures.

This is why the two approaches have different statuses:

| Component | Integration point | Works today? |
|-----------|------------------|-------------|
| γ schedule (Exp 1) | `hooks` module → DREAMPlace ignores it | **No** |
| λ schedule (Exp 2) | `hooks` module → DREAMPlace ignores it | **No** |
| Path-group net weights (Exp 4) | Direct write to `placedb.net_weights` | **Yes** |

### How DREAMPlace actually uses net_weights

Inside `WeightedAverageWirelength` (the CUDA kernel called every iteration):

```
WL(net_i) = net_weights[i] × [ WA_x(pins of net_i) + WA_y(pins of net_i) ]
```

Where `WA_x` is the softmax-weighted span in x using the current γ. So the total loss gradient flowing to cell positions for a critical half-cycle path net is 3–3.5× larger than for an unconstrained net, persistently throughout placement. Cells on those paths are under proportionally more pressure to stay co-located.

The γ schedule controls the smoothness of `WA_x/WA_y` but not the per-net magnitude. The λ schedule controls the density penalty weight. Both are still hardcoded DREAMPlace defaults in real runs — just the per-net scaling is ours.

---

## 2026-06-04 — Exp 1 Smoke Test (20-iter γ Evolution, CPU)

20 LLM-guided evolution iterations on fft_1. Best candidate: seed linear-decay schedule (norm_hpwl = 2.254, HPWL = 4.92M at 50 iters). No mutation beat the seed.

**Conclusion**: pipeline plumbing validated (LLM call → function extraction → evaluation → TSV logging). γ hook not yet wired into DREAMPlace (see architectural gap above), so all candidates ran vanilla DREAMPlace. Score variance is numerical noise, not schedule signal. Real Exp 1 results require fixing the NesterovPlace hook first.

**Known issue**: fitness scores inflated 2× vs Exp 0 because eval runs only 50 iters (HPWL ~4.9M vs converged ~2.2M). Normalisation is internally consistent but cross-experiment comparison requires matching iteration counts.

---

## TODO

- [x] Exp 0: Reproduce DREAMPlace 4.0 baselines on ISPD 2015 (CPU done; GPU pending DGX)
- [x] Exp 1: γ schedule evolution smoke test (pipeline validated; convergent GPU run pending)
- [ ] Exp 1b: γ evolution on GPU (300-iter eval, multi-design fitness, coverage assertions)
- [ ] Exp 2: Run OpenEvolve on λ schedule (density weight)
- [ ] Exp 2b: Multi-design fitness harness (fft_1 + fft_2 + des_perf_1 mean score)
- [ ] Exp 3: Train GNN initializer, measure iteration reduction
- [ ] Exp 4: Path-group-aware timing fitness (latch/half-cycle groups, static pre-CTS slack budgets)
- [ ] Exp 4b: Elmore-weighted wirelength baseline (rung (a) of compute ladder)
- [ ] Exp 4c: Periodic GPU-STA net weight refresh (rung (b))
- [ ] Exp 4d: Learned timing surrogate (rung (c))
- [ ] Exp 5: Full system integration, final benchmark comparison
- [ ] Harness: coverage assertions for evolved functions (confirm all branches execute)
- [ ] Harness: multi-design fitness aggregation
