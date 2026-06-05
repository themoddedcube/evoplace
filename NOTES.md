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

---

## 2026-06-04 — Empirical Lessons Imported from the Macro Placement Challenge Campaign (Klein-4)

Cross-pollination from the NG45/ORFS measurement campaign in
`autoresearch-macro-place-challenge-2026` (see its `docs/paper/NOTES.md` for
raw data). Three findings bear directly on EvoPlace's thesis and design.

### 1. Direct evidence the proxy/timing gap is real — and can invert

Same design (mempool_tile NG45), same ORFS flow, same machine, three macro
placements:

| placement | proxy cost | post-route WNS | post-route TNS |
|---|---|---|---|
| engine cold | 0.7092 | **-1.908** | **-12,633** |
| RTL-MP baseline | n/a | -1.946 | -13,938 |
| engine warm (best proxy) | **0.6666** | -2.089 (worst) | -14,484 |

The best-proxy placement had the WORST timing — a measured rank inversion,
not just decorrelation (campaign-wide Kendall tau ≈ 0.05 proxy↔WNS).
Validates EvoPlace's core premise (optimize TNS, not HPWL/proxy) with a
clean controlled triple.

### 2. Where timing dies is structural — the loss must see clock/path structure

mempool_tile's -1.9ns is dominated by half-cycle latch paths (level-sensitive
latches on inverted clock = 2ns budget at 4ns period) with ~1.5ns of clock
insertion eaten from the window. No wirelength-shaped objective can see this;
a timing surrogate that knows path budgets (even a static, pre-CTS estimate
of per-endpoint required times) would. Suggests an EvoPlace fitness/feature:
per-path-group slack estimates, not just global TNS — latch/half-cycle path
groups need separate weighting.

### 3. Compute budget for timing-in-the-loss (the practical ladder)

Cost concern is real but bounded; precedent is DREAMPlace 4.0 itself
(GPU-STA net weighting, ~2-4x per affected iteration). The ladder, cheapest
first:
1. Elmore/criticality-weighted wirelength (near-free, differentiable);
2. periodic GPU-STA refresh every N iters updating net weights
   (DREAMPlace 4.0 style — already in the EvoPlace backbone);
3. learned timing predictors as fitness surrogates.
Architecture trick from the challenge engine's funnel: keep broad cheap
search timing-blind (scouts), apply the timing term only to top-k survivors
(extend phase) — pays surrogate cost on ~20% of compute.

### 4. Cautionary prior art for the evolutionary loop itself

The Klein-4 campaign's documented negative result: an OpenEvolve run over CD
hyperparameters came out 13% WORSE after 16 generations — root causes were
(a) single-benchmark fitness → overfit, (b) evolved parameters that were
dead code paths, (c) too-narrow search space on the parameters that mattered.
EvoPlace mitigations: multi-design fitness from day one; assert evolved
components are actually exercised (coverage check in the harness); evolve
schedules/functions, not just scalar knobs.

### Flow-measurement gotchas worth inheriting (ORFS, NG45)

- "Repair converged" != signoff: rsz repair vs estimated parasitics can show
  0.000 WNS while extracted post-route STA shows -1.9. Only 6_finish.rpt counts.
- ORFS upstream CI itself accepts negative WNS on ariane133/136
  (rules-base.json: -0.464/-0.300) — calibrate fitness expectations per design.
- Genus-netlist designs keep literal-backslash escaped names in ODB
  (macro_mem\[0\].i_ram) — name-matching code must handle both forms.
- kepler-formal (LEC) binary requires AVX-512 — SIGILLs on Zen 3 hosts;
  LEC_CHECK=0 is the surgical disable (verification-only, netlist-invariant).

---

## 2026-06-04 — GPU bring-up (RTX 3060) + recovery of lost integration

Machine move CPU→GPU exposed that the entire DREAMPlace integration only
existed as uncommitted local state on the old machine and was lost:

1. **Hook-patched PlaceObj.py** — not in the fork, not in any branch, no
   patch file. Recreated against the dreamplace_ext/hooks.py contract and
   COMMITTED this time (fork branch `evoplace-hooks`, archived copy at
   dreamplace_ext/patches/). Evolved γ is dimensionless [0.01, 50], scaled
   by (bin_size_x + bin_size_y) inside update_gamma. λ hook wraps
   update_density_weight. init_positions / timing_loss hooks remain unwired
   (Exp 3/4). This was exactly Klein-4 failure mode (b): "evolved parameters
   that were dead code paths" — without the patch every candidate evaluated
   identically and no test caught it.
2. **run_placement.py real path could never have run as committed**
   (abstract BasicPlace, dict-for-Params, metrics list indexed as dict).
   Rewritten: Params.load + NonLinearPlace + nested-metrics extraction.
3. **Build fixes for modern toolchain**: cub CUB_NS_PREFIX wrapping breaks
   CCCL 2.x (CUDA 12.6) → global include + namespace alias; np.string_
   removed in NumPy 2.0 → np.bytes_. Both committed to the fork.

**GPU baselines (fft_1/fft_2, seed 42, stop_overflow 0.07 + legalization)**:
2.180e6 @ ovf 0.082 (29.4s) / 1.921e6 @ ovf 0.070 (12.2s). The CPU-era
"norm 2.268 @ overflow 0.65, 50-iter" numbers reproduce exactly on GPU at
50 iters (2.275 / 0.65) — CPU runs were real, just truncated.

**Cascade was structurally broken**: stage thresholds (2.0/1.3) normalized
truncated-run HPWL against the CONVERGED baseline; the default schedule
itself lands at 2.28x after 50 iters → everything culled. Fix: per-stage
baselines [50/300/full]: fft_1 4.96e6/5.42e6/2.18e6.

**Seed program was not the default it claimed to be**: linear 8→0.5 vs
DREAMPlace's overflow-driven 4.0×10^((ovfl−0.1)·20/9−1) (~40→0.4). Seed now
reproduces the true default; verified through the real cascade at
norm_hpwl = 0.999. Evolution starts at parity, not from a 2x handicap.

Exp 1 (20 iters, claude-code-cli backend) launched. Cascade cost/candidate:
~5s cull at stage 0, ~70s full pass.
