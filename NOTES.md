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

## 2026-06-05 — DGX Spark (GB10) Bring-up + Exp 1 Relaunch

Followed SPARK_SETUP.md. Deviations worth recording:

1. **libfl-dev was missing from the apt list** — Limbo's bison parsers need
   `FlexLexer.h` (FLEX_INCLUDE_DIR NOTFOUND at configure). Added to §2 deps.
2. **CUDA 13.0 broke two CMake assumptions** (both committed to evoplace-hooks):
   the sm_120 manual-append condition `major>=12 AND minor>=8` is false for
   13.0, and the default gencode list still contained sm_60/61/70, which
   nvcc 13 dropped. Fixed: CUDA>=13 gets archs {7.5..9.0} + sm_120.
3. **HeteroSTA prebuilt release is x86-64 only** → timing_heterosta cannot
   link on aarch64. Skipped on non-x86_64 (only needed for Exp 4 timing;
   revisit before Exp 4 on this machine).
4. PyTorch 2.12.0+cu130 aarch64 wheel works out of the box: GB10, cc (12,1).

**Tests**: 114/114. **GB10 baselines (seed 42)**: fft_1 2.1827e6 @ ovf 0.083
(5.5s), fft_2 1.9509e6 @ ovf 0.070 (2.2s) — ~5x faster than the 3060, HPWL
within 0.1%/1.6% of the 3060 values. Stage baselines re-measured (within
0.7% of 3060); both dicts in evaluator_wrapper.py updated.

**Sanity gate**: seed norm_hpwl = 0.99932 ✓. Added a differential liveness
probe beyond the guide: a constant γ=0.01 program gets culled by the cascade
at stage 1 while the seed passes at ~1.0 — hooks demonstrably steer the
placement. Recommend making this probe a standing part of the gate.

### Exp 1 relaunch (200 iters, claude-code-cli) — NOISE WARNING

Status at iteration ~93/200: 84/93 candidates culled by cascade. Finite
scores (norm_hpwl): 0.9956, 0.9992, 0.9992, 0.9998, 1.0001 (seed), 1.0035,
1.0065, 1.0227, 1.0351.

**Most "improvements" so far are indistinguishable from noise.** With the
single-seed noise floor at σ ≈ 0.15% (measured on the 3060 campaign), every
survivor except 0.9956 sits within ~2σ of the seed. The 0.9992 "new bests"
that the hill-climber celebrated are 0.5σ events — exactly the failure mode
where evolution chases seed luck instead of schedule quality. Only the
0.9956 candidate (−0.44%, ~3σ) is a plausibly real effect, and even that is
one sample.

**Protocol going forward (do not skip):**
1. Treat single-seed rankings as candidate *generation* only, never as
   results. No claimed improvement below ~0.45% (3σ) from a single seed.
2. After every evolution run: `scripts/multiseed_rerank.py` — paired
   candidate/default ratios on identical (bench, seed) pairs, 5 seeds ×
   {fft_1, fft_2}. Pairing cancels per-seed placement luck; the seed program
   rides along as a calibration row (its ratio must be ~1.000).
3. Visual comparisons (make_comparison_gif.py, plot_interval patch) are
   prepared but DEFERRED to the end of the campaign — no GPU contention with
   scoring runs, and no GIFs of effects we haven't confirmed are real.
4. Medium-term fix is multi-seed fitness *inside* the evolution loop (the
   thing this 128 GB box was bought for) — score = mean over 3-5 seeds,
   which drops the noise floor to ~0.07% and makes 0.2% effects climbable.

## 2026-06-05 — Related Work: yaoxufeng/EvoPlace (arXiv 2504.17801) — NAME COLLISION

User found github.com/yaoxufeng/EvoPlace — "Evolution of Optimization
Algorithms for Global Placement via LLMs". Same name, same base placer
(DREAMPlace 4.1), same core idea. MUST cite; likely must rename our project
for the paper (theirs appears to predate us publicly).

What they evolve: initialization / preconditioner / optimizer (C++ level).
NOT γ/λ schedules — our Exp 1/2 territory is untouched. Their headline:
case-by-case 5.05% (MMS) / 5.29% (ISPD2005) / 8.30% (ISPD2019) HPWL
reduction; initialization contributes the most (up to 17.7% adaptec3).

Implications for us:
1. **Exp 3 (GNN warm-init) gains priority** — independent evidence that
   init is the highest-yield component.
2. **Their scaling law**: HPWL gain vs #generated candidates is
   LOGARITHMIC; breadth (≥1000 offline candidates + diversity-aware
   selection by performance + negative cosine similarity, then UCB +
   self-reflection evolution) beats long hill-climbs. Our 200-iter
   single-lineage run is the opposite allocation — consider a
   generate-many/select-diverse stage for Exp 2.
3. **They are single-seed** and admit ~1% cross-environment noise — the
   same trap we're guarding against. Our paired multi-seed protocol is a
   methodological differentiator worth a paragraph in the paper.
4. Benchmark context: their gains are on macro-heavy ISPD2005/MMS with more
   headroom; ISPD2015 fft_* are small std-cell designs. Our ~0.4% is not
   directly comparable. Consider adding ISPD2005/MMS (bookshelf, supported
   by DREAMPlace) for an apples-to-apples table.
5. Their generalization collapse (5.05% → 0.51% when one algorithm must
   serve all cases) is the same Klein-4 overfit failure mode we documented —
   multi-design fitness stays mandatory.

## 2026-06-05 — yaoxufeng/EvoPlace deep-dive (repo + paper) — synthesis

Full reports from repo dissection + paper deep-read (arXiv 2504.17801 v1;
GPT-4o `2024-08-06`, 60×2080Ti + 150×3090, ≥1000 candidates/component,
1000 trials/case). Key facts with bearing on our campaign:

**Where their gains live.** Init ≫ optimizer > preconditioner, and the
big wins are macro-position initialization on macro-heavy MMS/ISPD2005
(adaptec1/3/4: 11–18%). On clean/std-cell-like cases their gains are at
the noise floor: newblue5 0.03%, bigblue4 0.20%, adaptec5 0.59%. Their
~5% suite averages are carried by ~6 of 16 cases. They are SINGLE-SEED
(`random_seed: 1002`, deterministic_flag) with admitted ~1% environment
noise → their sub-2% rows are statistically zero. Our fft_1 ≤0.44% result
is consistent with their std-cell regime, not an anomaly of our setup.

**Generalization collapse**: case-by-case 5.05% → generalized 0.51% (MMS),
5.29% → 2.85% (ISPD2005). DECISION NEEDED: per-case schedules vs
generalizable schedules — expected gain differs ~10×. Their generalized
ISPD2005 2.85% is roughly the ceiling for our (generalizing) approach.

**Inference scaling is logarithmic** in candidate count (Fig 10c). Long
hill-climbs are the wrong compute allocation; breadth-first generation +
diversity-aware selection (perf + negative cosine sim of code embeddings,
k-clique-densest-subgraph; then UCB w/ self-reflection) is theirs. Note:
α, β, λ(UCB), temperature all unreported — reproducibility holes.

**Repo reality check**: partial release — prompt/ dir and llm_*_evolution
drivers OMITTED ("future version"). Evaluation is `cp` candidate over
DREAMPlace source + subprocess + regex wHPWL parse; timeout→HPWL=1e16.
Our hook injection + cascade is structurally ahead. Their edge is PROMPT
CRAFT, not orchestration.

**Adoptable (ranked, from their prompts/winners):**
1. placedb-statistics catalog in the prompt with concrete example values
   (lengths, areas, bin sizes) + "use statistical reductions, not raw
   arrays" guidance → ground mutations in the actual circuit. Add fft_1
   overflow trajectory facts to config.yaml system_message.
2. CoT analysis step before mutation (their @@Macro-Init-Ana@@ splice);
   with Claude 4.x this is one <analysis> block in the user template.
3. Multi-design suite fitness: their score = mean(1 − cur/baseline) over
   the suite. We evolve on fft_1 alone → add fft_2 (baselines already in
   evaluator_wrapper) — also our Klein-4 mitigation.
4. Mandatory "Key improvement points summary" docstring in candidates
   (auditable archive + forced reasoning).
5. Retrieval-augmented mutation: sample one reference schedule (cosine /
   exponential / overflow-PID / piecewise, as CODE not names) per prompt
   (their optimizers/*.txt pool, 72 files).
6. Winning-code motifs to seed the reference pool: density/area-ratio
   anchoring, net-weight-std-scaled noise, statistical reductions,
   BB/Lipschitz adaptive steps, SA cooling.

**From VeoPlace (arXiv 2603.28733, Gemini 2.5 Flash, no fine-tune):**
3-seed mean±SE reporting (adopt); 512×512 placement renders as VLM
feedback (k-means connectivity coloring) — for us: feed placement image +
γ/overflow trajectory PLOT into the reflection step; weak soft-constraints
beat strong (λ_A=0.001) — prior for Exp 2 λ schedules; Top-Stratified
parent selection (rank-softmax over clusters, τ≈0.43) as cheap A/B vs
MAP-Elites sampling.

**Strategic conclusion**: we are attacking their weakest lever (schedule/
optimizer) on their weakest design type (std-cell) in their weakest
setting (generalization). Exp 1 completes as a boundary-result; priority
shifts to Exp 3 (init — their strongest lever, and ours can be net-driven
GNN warm-init rather than their statistical heuristics), Exp 2 restructured
as breadth-first + multi-design fitness + vision-in-the-loop reflection.
Also: PAPER must cite 2504.17801 + 2603.28733; project rename likely.

## 2026-06-05 — Deep research: "what makes a better placer" (RESEARCH.md)

Ran a 6-angle, 108-agent deep-research sweep with 3-vote adversarial
verification (117 claims → 25 verified → 20 confirmed, 5 killed). Full
synthesis in RESEARCH.md. Bottom line: schedule/optimizer/throughput work
doesn't move QoR anywhere in the literature (matches our <0.5% result);
verified levers are (1) direct differentiable routability/timing objectives
in the GP loop (RoutePlacer 13-16% overflow multi-seed; differentiable STA
"free timing"), (2) structure injection (DG-RePlAce), (3) init (~1% honest
avg, macro-heavy only). RL macro placement confirmed non-replicating.
Single-seed reporting confirmed as the field-wide hazard — our protocol is
ahead of practice. Recommended next lever: in-loop differentiable
routability (fits the timing_loss-style hook shape; evolvable penalty form;
requires ISPD2011/DAC2012 benchmarks). Spark-unique-capability axis (d) and
legalization/DP gap remain unevidenced — flagged as open questions.

## 2026-06-05 — Exp 1 FINAL: multi-seed re-rank verdict (boundary result confirmed)

Exp 1 completed: 200 iterations, 14 cascade survivors, single-seed best
0.9956 (candidate_0117). Paired multi-seed re-rank (5 seeds × fft_1/fft_2,
ratios vs default at identical (bench, seed)):

  1. candidate_0117  0.99685 ±0.00273  → +0.315% (REAL: ~3.7× SEM, but tiny)
  2. candidate_0090  1.00014 ±0.00851  → noise (was −0.28% single-seed)
  3. seed_program    1.00057 ±0.00735  → calibration ✓ (protocol validated)
  4. candidate_0006  1.02244            → 2.2% WORSE
  5. candidate_0007  1.06626            → 6.6% WORSE
  6. candidate_0002  1.07746            → 7.7% WORSE

Three of five single-seed "top" candidates were actively bad schedules that
got lucky once — single-seed ranking at the tail was anti-correlated with
truth, not merely imprecise. CONCLUSION (campaign boundary result): LLM
schedule evolution on std-cell ISPD2015 finds at most ~0.3% real HPWL
improvement (candidate_0117: overflow-driven exponential γ with progress
envelope), far below useful headroom. Consistent with RESEARCH.md verified
findings. Campaign closes; next lever per RESEARCH.md is in-loop
differentiable routability. Full table: experiments/exp01_wl_smoothing/
multiseed_rerank.tsv.

## 2026-06-05 — CORRECTION to re-rank interpretation (data provenance)

results.tsv contains duplicate iteration indices (e.g. iter 2 appears with
norm 0.998 AND N/A) — index collisions between TSV records and candidate
files. Consequence: re-rank rows 4-6 (candidates 0002/0006/0007, measured
2.2-7.7% worse) were selected from stale/collided records and their files
are cascade-REJECTED programs, NOT single-seed top candidates. The earlier
claim "3 of 5 top candidates were actively bad / tail anti-correlated" is
therefore an overstatement — retracted.

Corrected findings (unchanged measurements, corrected provenance):
- Single-seed BEST was candidate_0090 (0.99563); multi-seed it is exactly
  noise (1.00014 ±0.0085). Single-seed runner-up candidate_0117 (0.99716)
  is the real one: 0.99685 ±0.0027 (+0.315%). → rank-1↔2 inversion at the
  noise floor; ordering below ~0.45% single-seed is meaningless.
- seed_program calibration 1.00057 ✓.
- Rows 4-6 function as negative controls: protocol correctly measures
  known-bad schedules as 2-8% worse with tight CIs.
Boundary conclusion unchanged: best real evolved-schedule gain ≈ +0.3%.

## 2026-06-05 — Exp 2 pre-evolution FINDING: unconditional λ ramp beats default

Writing the Exp 2 seed exposed an API constraint: the λ hook receives
(iteration, overflow, overflow_history, gradient_norm, current_lambda) but
NOT delta_hpwl/ref_hpwl, so DREAMPlace's default density-weight update
cannot be reproduced exactly. The seed approximates the default's
HPWL-improving branch applied UNCONDITIONALLY:
    λ_{k+1} = λ_k · 1.05 · max(0.9999^k, 0.98)
i.e. the default minus its guard branch (which shrinks the multiplier when
HPWL worsens between evaluations).

Sanity gate flagged non-parity at 7σ (0.9896 vs expected 1.0±0.01). Paired
multi-seed verification (5 seeds × fft_1/fft_2, identical final overflow
both sides, converged flags matched per design):

  fft_1: ratios 0.9864–0.9902 (5/5 better), ≈ −1.0%
  fft_2: ratios 0.9026–0.9212 (5/5 better), ≈ −8.7% (!)
  pooled mean 0.9511 ± 0.0128 SEM

INTERPRETATION (pending robustness check on matrix_mult_1/des_perf_1):
DREAMPlace's HPWL-feedback guard in update_density_weight appears to
actively hurt these std-cell designs — slowing the λ ramp mid-flight takes
longer trajectories into worse minima. The unconditional ramp reaches the
same density quality (same overflow) at substantially lower wirelength.
Effect is design-heterogeneous (1% vs 9%) — classic in this field; do not
extrapolate beyond measured designs.

Provenance note: this is NOT an evolved result — it is an ablation of the
default's guard branch, found accidentally via the hook-API constraint and
caught by the sanity gate. The protocol (gate → 7σ flag → paired
verification) worked exactly as designed.

**Robustness check (same protocol, 2 seeds each):** matrix_mult_1 ratios
0.98294/0.98277 (−1.7%), des_perf_1 0.99053/0.99034 (−1.0%), all at matched
overflow. Total: 14/14 paired wins across 4 designs, −1% to −8.7%. Unlike
the per-case-evolved results in the literature, this single fixed schedule
GENERALIZES across every design tested. Guard-branch ablation is now the
campaign's headline empirical finding; Exp 2 evolution launches from this
seed.

## 2026-06-05 — Exp 2 reward hack caught live; overflow gate added

9 iterations into Exp 2, a mutant scored norm_hpwl 0.9785 ("new best") at
FINAL OVERFLOW 0.352 — five times the stop criterion. Classic λ-schedule
exploit: suppress the density ramp, cells stay clustered, HPWL is
artificially low, and an HPWL-only fitness rewards it. γ evolution couldn't
express this failure mode (γ doesn't control spreading); λ evolution can.

Fix: evaluator_wrapper now rejects any candidate whose final mean overflow
exceeds 0.12 (defaults end at 0.07–0.085 on these designs) — score −inf,
stage "overflow_gate". The contaminated 9-iteration run is archived at
evolution_runs_hacked_archive/; Exp 2 relaunched from scratch with the
gate active. Tests still 114/114.

Lesson for the protocol list: every objective the schedule can influence
needs an explicit gate or the search WILL find the loophole — fitness
specification is part of the noise discipline.

## 2026-06-05 — Operational incident: orphaned ungated run, interleaved campaigns

The overflow-gate "fix" initially appeared to fail: 30+ minutes after the
gated relaunch, top candidates were again scoring 0.96-0.97 at overflow
0.35. Root cause was operational, not logic: the kill before relaunch
targeted the BASH WRAPPER PID, not the python child. The original ungated
run survived as an orphan (reparented to init) and kept evolving reward
hacks, while the gated relaunch ran concurrently — two evolution loops
interleaving writes into the same log, results.tsv, and candidate files,
and contending for the GPU (which perturbs runtime-derived MAP-Elites
features in both).

Detected by direct gate test (fresh process correctly returned -inf for
the hacked best_program) + process listing showing both PIDs with their
start times. Resolution: kill -9 both, archive contaminated output to
experiments/exp02_density_schedule/evolution_runs_contaminated_interleaved/,
relaunch once as a properly tracked process. Verified in-loop: an
under-spread candidate (overflow 0.266) passed the HPWL cascade and was
gated to -inf.

Operational rules added to the discipline list:
1. Kill the PYTHON process, verify with pgrep by command line, not by
   remembered PID — wrappers die, children get orphaned.
2. After any fix-and-relaunch, verify the fix IN-LOOP (first gated
   elimination observed), not just in a fresh-process unit test.
3. One campaign per log file / output dir, ever. Truncating a log a live
   process holds open does not stop it writing.

## 2026-06-05 — Exp 2 FINAL: gated λ evolution finds nothing beyond the seed

Clean gated run completed 200/200. Outcome: 150/200 candidates scored -inf
(cascade + overflow gate), 49 survivors all WORSE than the seed
(1.09-3.23), seed (0.9890 on fft_1) remains best — best_program.py is the
unmodified seed. Contrast with the contaminated ungated run, which kept
"improving" via under-spread placements: with legality enforced, λ-space
beyond the unconditional ramp offered the search nothing but hacks.

Verdict: Exp 2's contribution is the guard-branch ablation finding (seed,
+1 to +8.7% vs default at matched overflow, 14/14 paired seeds, 4 designs);
LLM evolution on top of it: zero. Combined campaign conclusion: schedule
*search* is dead in both γ and λ spaces; schedule *auditing* produced the
only real win. No multi-seed re-rank needed (no candidate within 60σ of
the seed).

## 2026-06-06 — Gallery GIF timing unified; Honesty section now leads with the audit number

Presentation-only fixes, no new measurements. (1) The README gallery GIFs
were captured at different --interval values (fft_1/fft_2: 25, superblue12:
50) but all rendered at 10 fps, so iteration counters advanced at 250 vs
500 iters/sec and loop periods ranged 4.4–10.3 s — visually confusing side
by side. New scripts/retime_gifs.py rewrites frame durations to a common
250 iters/sec and pads each final-frame hold to a shared 10.2 s loop
period, so all nine GIFs advance and restart in sync (idempotent; timing
derived from the recorded capture intervals, pixel content untouched —
verified frame-by-frame: counters 0000/0400/0600/1051 at the expected
wall-clock times). make_comparison_gif.py now defaults fps to
--iters-per-sec 250 / --interval so regenerated figures can't drift;
--fps remains as an explicit override. docs/RUNNING.md documents both.
(2) README "Honesty, up front" stated only the evolution bound (+0.315% ±
0.09%) and omitted the campaign's largest number; it now states both with
the distinction preserved: evolution → +0.315% at best, *auditing* the
default's λ guard branch → 1–9% at matched density (14/14 paired seeds,
4 designs). No claims-ledger change — both numbers were already confirmed
(2026-06-05 entries).

Verification gotcha worth recording: PIL's ImageSequence.Iterator yields
the SAME mutable Image object each step, so list(Iterator) gives N
references to the final frame — convert() inside the loop, or you'll
"verify" the wrong thing (this produced a false alarm during checking;
retime_gifs.py itself converts per-frame and was always correct).

## 2026-06-06 — fft_2 electrostatics GIFs generated; gallery layout made uniform

The λ audit-finding section was the only gallery entry without the
density/potential/field surface row. Regenerated it with --fields all
(seed program experiments/exp02_density_schedule/evolution_runs/
best_program.py, --hook lambda, fft_2, seed 42, interval 25; command now
recorded in docs/RUNNING.md). The rerun independently reproduced the
audit result: evolved 1.7434e+06 vs default 1.9178e+06 → −9.09% (prior
render: −9.46%; both inside the documented per-seed ratio range
0.9026–0.9212 from the 2026-06-05 entry — run-to-run GPU nondeterminism,
no claims change). All 12 GIFs re-synced to the common 250 iters/sec /
10.2 s loop via retime_gifs.py; mid-frame counters verified (iteration
0450 at frame 18 × 100 ms ✓). README gallery now uniform: every design
section is heading + description + main GIF + bare field-surface table
(the separate "Electrostatics in motion" heading is gone).

## 2026-06-06 — Sign convention unified: GIF banners now show signed ΔHPWL (negative = better)

The comparison-GIF suptitle computed Δ as (1 − evolved/default)·100
(improvement-positive, so the λ ablation read "+9.46%") while the README
text reported signed wirelength change ("−9%"). make_comparison_gif.py now
prints Δ = (evolved/default − 1)·100 — negative = better — and both
comparison GIFs were regenerated. Reruns (GPU nondeterminism, no claims
change): fft_2 λ ablation −9.34% (range across renders −9.09/−9.34/−9.46%,
all inside the documented 0.9026–0.9212 per-seed band); fft_1 γ candidate
−0.42% single-seed (multi-seed claim stays −0.315% ± 0.09%, σ ≈ 0.15%
noise floor). README γ mentions aligned to the same convention (gallery
caption now "−0.315% ± 0.09% HPWL"; prose uses unsigned
"reduction"/"gain" wording). All GIFs re-synced to the common 250
iters/sec / 10.2 s loop.
