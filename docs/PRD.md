# PRD — Contract-Driven GPU Placer (working name: "Covenant")

**Author**: Chaithu Talasila · **Date**: 2026-06-05 · **Status**: Draft for new-repo kickoff
**Provenance**: Distilled from the EvoPlace campaign ([NOTES.md](../NOTES.md)), the adversarially-verified literature review ([RESEARCH.md](RESEARCH.md)), and the campaign paper ([paper/paper.pdf](../paper/paper.pdf)).

---

## 1. Vision & Positioning

**An open-source GPU analytical placer whose product promise is constraint satisfaction with honest reporting — not metric optimization.**

Every placer today, open or commercial, optimizes a metric (wirelength, weighted TNS) and leaves the engineer to discover after routing whether timing closed. We invert the interface: the engineer declares timing constraints up front; the placer treats them as **requirements, not objectives** — trading wirelength for constraint satisfaction at placement-level STA, and when targets are infeasible, reporting *exactly which paths cannot close, by how much, and what was traded in the attempt*.

> **The contract: timing-met-or-explain.** We do not use the word "guarantee." Placement alone cannot ensure post-route timing — routing, CTS, and sizing happen downstream, and placement-level STA is an estimate. What we contract is: meet the constraints at our STA model or hand you a ranked, actionable infeasibility report. Honesty about that boundary is the brand.

**Positioning:**

| | DREAMPlace | Xplace | NVIDIA C3PO | **This placer** |
|---|---|---|---|---|
| GPU analytical GP | ✅ | ✅ (faster) | ✅ | ✅ (DREAMPlace fork) |
| Timing in the loop | net weighting (4.0) | Xplace-Timing | ✅ differentiable | ✅ differentiable |
| Routability in the loop | ❌ | in-loop router | ✅ | ✅ |
| **Constraint-contract semantics** | ❌ | ❌ | ❌ | ✅ **core promise** |
| Variance-qualified results | ❌ | ❌ | unverifiable (closed) | ✅ built into the tool |
| Open source | ✅ | ✅ | ❌ | ✅ |

C3PO (ASP-DAC 2026) validates the technical direction — concurrent differentiable timing/routability/WL extends DREAMPlace, exactly our architecture — but it is closed, its QoR numbers are unverifiable, and it has no contract semantics. We are not racing NVIDIA on kernels; we differentiate on **the interface (contract), the epistemics (variance-qualified everything), and openness**.

---

## 2. Why Now — Evidence Base

Every claim below carries its verification status from our campaign or the 3-vote adversarial literature review.

1. **Direct differentiable objectives are the validated QoR lever.** RoutePlacer (KDD'24): differentiable routability penalty → 13–16% routing-overflow reduction at flat WL, *5-seed mean±std* (rare in this field). Differentiable STA (DAC'22): timing as a smoothed global objective costs ~nothing in WL, vs. net-weighting's ~4.3% WL penalty. [verified 3-0]
2. **Schedule/optimizer tuning is a dead end** — our own measured bound: γ-schedule headroom ≤ ~0.3% on std-cell ISPD2015 (200-iteration LLM evolution campaign, paired multi-seed confirmation). Consistent with arXiv 2504.17801's component ordering. [our measurement]
3. **The control heuristics of existing placers are unaudited and contain real defects.** Our λ guard-branch ablation: removing DREAMPlace's HPWL-feedback guard from the density-weight update wins **1–9% HPWL at matched final overflow, 14/14 paired seeds across 4 designs** — a generalizing improvement found by *auditing*, not evolving. Nobody has systematically done this. [our measurement, 2026-06-05]
4. **Single-seed evaluation is the field-wide credibility gap.** Nearly every published >2% placement gain is single-seed; the RL-placement saga (gains traced to undisclosed initialization) is the cautionary tale. Our noise-calibrated protocol (measured σ, liveness gates, paired multi-seed with calibration rows) caught a rank inversion and a live reward-hack in our own campaign. Baking it into the tool is both a differentiator and a moat of trust. [verified 3-0 + our experience]
5. **RL policies and kernel-speedup work are not QoR levers** (verified 3-0) — excluded by design.

---

## 3. Users & Use Cases

- **U1 — The closure-focused engineer.** Has hard clock targets and a deadline; prefers predictable timing closure over 2% better wirelength. Workflow: write SDC → run placer in contract mode → get either a timing-clean placement or a ranked infeasibility report to take back to the architect.
- **U2 — The EDA researcher.** Needs a baseline placer whose reported numbers are statistically defensible and a benchmarking harness that produces CIs by default. Today they hand-roll seeds or (usually) don't.
- **U3 — The ML-for-EDA practitioner.** Needs clean injection points (objectives, schedules, init) to test learned components against an *audited* baseline — inherited directly from EvoPlace's hook design.

---

## 4. Product Requirements

### Functional

**FR1 — Timing contract (the headline feature).**
- Ingest an SDC subset: clock definitions, input/output delays, false paths, multicycle paths. (Full SDC is out of scope; the subset must be documented precisely.)
- Placement-level STA implemented as a **differentiable torch DAG** (LSE-smoothed min/max, DAC'22 formulation) — runs on any GPU incl. aarch64 (deliberately avoids the x86-only HeteroSTA binary), and doubles as the in-loop gradient source.
- **Contract mode** (lexicographic objective): satisfy constraints first, minimize wirelength second. Implementation direction: constraint-violation penalty with escalating weight + final verification pass, not a fixed weighted sum.
- **Infeasibility report**: when targets can't be met, emit ranked unclosable endpoints with worst slack, dominant path segments (cell/net delay split), and the WL that was sacrificed in the attempt. This report is a first-class output artifact (JSON + human-readable), not a log line.
- Honest-boundary reporting: every timing number labeled as placement-level estimate; correlation against OpenSTA on routed reference designs published in the docs.

**FR2 — Routability objective.**
- Differentiable congestion penalty co-optimized in the GP loop. Phase 1: RUDY-gradient (cheap, no learned components). Phase 2 (optional): GNN penalty à la RoutePlacer if RUDY plateaus.
- Acceptance: ≥10% routing-overflow reduction at ≤1% WL cost on ISPD2011, 5-seed CIs.

**FR3 — Audited control loop.**
- Every adaptive heuristic inherited from DREAMPlace (λ update branches, preconditioner updates, divergence rollback, stop criteria) gets a paired multi-seed ablation; results ship in the repo as `AUDIT.md`.
- The λ guard-branch fix is pre-seeded finding #1 (evidence already in EvoPlace NOTES.md).
- Defaults are chosen by audit evidence, never by upstream inertia.

**FR4 — Noise-calibrated benchmarking, built in.**
- `placer bench` subcommand: paired multi-seed comparison (candidate vs. baseline on identical (design, seed) pairs), calibration row, measured noise floor, CI on every number.
- **Policy: no single-seed numbers anywhere** — not in docs, not in README, not in tool output. Single-run mode prints its result with an explicit "single seed — not a claim" tag.
- Fitness/objective gates from day one (the Exp 2 lesson: an HPWL-only fitness was reward-hacked by under-spreading within 9 iterations; every metric a component can influence needs an explicit gate).

**FR5 — Extension layer.**
- EvoPlace-style hook registry preserved: schedules, initialization, additional differentiable objectives, all injectable as Python callables without rebuilds.
- The frozen-evaluator discipline: evaluation harness files are never modified during experiments.

### Non-Functional

- **Platforms**: CUDA 12.x/13.x; sm_80+ including GB10/aarch64 (CUDA 13 CMake fixes already exist on our DREAMPlace fork's `evoplace-hooks` branch).
- **Benchmarks**: ICCAD2015 (timing, has SDC), ISPD2011/DAC2012 (routability), ISPD2015 (WL regression suite).
- **Licensing**: DREAMPlace is BSD-3 — fork-friendly; audit transitive deps (OpenTimer is MIT; flute/NTUplace binaries need review) before first release.
- **Reproducibility**: pinned environments; every README number regenerable by one script; CI smoke on stub mode (no GPU).

---

## 5. Roadmap — Three Pillars

Pillars are independently shippable; each ends in a publishable, variance-qualified artifact.

### Pillar 1 — Audit & fix the control loop (~weeks)
Port the EvoPlace harness (evaluator, hooks, multiseed_rerank, sanity gates) onto a fresh DREAMPlace fork. Build the ablation matrix of every adaptive heuristic; run paired multi-seed on each.
**M1**: an audited baseline configuration beating stock DREAMPlace with CIs on ≥4 designs (the λ fix alone already clears this bar). Artifact: `AUDIT.md` + reproducer.
**Kill criterion**: none — cheap and always informative.

### Pillar 2 — Differentiable routability (~1–2 months)
RUDY-gradient penalty in the GP loop; ISPD2011/DAC2012 benchmarks + overflow evaluation; optional: LLM-evolution harness searches the penalty's functional form (a search space with *verified* headroom, unlike schedules).
**M2**: ≥10% routing-overflow reduction at ≤1% WL cost, 5-seed CIs — open, variance-qualified RoutePlacer parity.
**Kill criterion**: <5% overflow effect after penalty-form search → publish the negative result, proceed to Pillar 3.

### Pillar 3 — The timing contract (~2–4 months)
Torch differentiable STA; SDC-subset ingest; contract mode (lexicographic escalation); infeasibility report; OpenSTA correlation study.
**M3**: on ICCAD2015, contract mode meets relaxed clock targets that stock DREAMPlace misses at equal density, and produces correct, actionable infeasibility reports on deliberately-infeasible targets.
**Kill criterion**: if placement-STA ↔ sign-off correlation is too weak for the contract to predict anything (measure this *first*, before building contract mode), the pillar pivots to "timing-driven mode without contract language."

---

## 6. Success Metrics

- **M1/M2/M3 quantitative gates** as defined above — all paired multi-seed with CIs; no exceptions.
- **Trust metric**: a third party reproduces the README's headline numbers from the provided script within stated CIs.
- **Adoption proxies**: external issues/PRs against the hook layer; citations of `AUDIT.md` findings.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Placement-STA ↔ post-route fidelity too weak | Measure correlation first (Pillar 3 gate); publish estimated-vs-routed deltas; contract language scoped to placement-level explicitly |
| C3PO ships first / better | Different product: open, contracted, audited. Their publication validates our architecture choices |
| Single-maintainer bandwidth | Pillars independently shippable; P1 alone is a complete contribution |
| Benchmark licensing (ICCAD2015 registration) | Free academic registration; document the path; keep ISPD suites as no-registration fallback |
| The audit finds nothing beyond λ | Already disproven — λ finding is in hand; even a null audit is a publishable negative result under our protocol |
| Reward hacks in new objectives | FR4 gate policy from day one; every objective ships with its gate |

## 8. What Carries Over From EvoPlace

| Asset | Reuse |
|---|---|
| `evaluator/run_placement.py` | The frozen-evaluator pattern + cascade evaluation (port, keep stage-matched baselines) |
| `dreamplace_ext/hooks.py` + PlaceObj patches | The injection layer (extend with objective-term hooks) |
| `scripts/multiseed_rerank.py` | Becomes the core of `placer bench` |
| Sanity-gate runbook (RUNNING.md) | Seed parity + differential liveness probes, verbatim |
| Overflow-gate lesson (NOTES 2026-06-05) | FR4 gate policy |
| DREAMPlace fork `evoplace-hooks` branch | CUDA 13/aarch64 build fixes, plot_interval, hook patches — fork from here |
| SPARK_SETUP.md | GB10 bring-up guide |
| NOTES.md discipline | Continuous dated lab notebook from day one |

## 9. Out of Scope

- RL placement policies (verified non-replicating).
- Initialization research as a pillar (crowded, ~1% ceiling; revisit only post-M3).
- Detailed-placement rewrite (use DREAMPlace's; the GP↔DP gap is a flagged open question, not a commitment).
- "Guaranteed timing" language — anywhere, ever.
- Full SDC support (documented subset only).

---

*Naming note: "Covenant" is a placeholder that captures the contract idea — pick the real name before the repo goes public, and check for collisions first (we learned this the hard way: EvoPlace collided with arXiv 2504.17801).*
