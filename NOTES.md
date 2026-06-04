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

## TODO

- [ ] Exp 0: Reproduce DREAMPlace 4.0 baselines on ISPD 2015 + ICCAD 2015
- [ ] Exp 1: Run OpenEvolve on γ schedule (WL smoothing)
- [ ] Exp 2: Run OpenEvolve on λ schedule (density weight)
- [ ] Exp 3: Train GNN initializer, measure iteration reduction
- [ ] Exp 4: Train timing surrogate MLP, integrate into loss
- [ ] Exp 5: Full system integration, final benchmark comparison
