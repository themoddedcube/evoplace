# What Would Make a Meaningfully Better Placer? (2026-06-05)

Deep-research synthesis: 6 search angles, 25 sources fetched, 117 claims
extracted, 25 adversarially verified (3-vote panels; 20 confirmed, 5 killed).
Question: what to change about DREAMPlace / modern placers to build the next
best placer, given a small team, an LLM-evolution harness, and a DGX Spark
(GB10, 128 GB unified memory).

## Headline

The replicated 2024–2026 finding mirrors our own campaign result: **schedule/
optimizer tuning and GPU-throughput work do not move QoR** (Xplace's 2-3×
speedup ⇒ ~0.3% HPWL; its neural-operator extension ⇒ ~0.1%; our γ/λ result
<0.5%). What does move QoR, with verified evidence:

1. **Direct differentiable objectives inside the GP loop** — make timing and
   routability first-class differentiable terms with exact gradients, not
   net-weighting/cell-inflation proxies.
2. **Injected domain structure** — dataflow/datapath constraints in the
   objective.
3. **Initialization** — confirmed again, but the honest multi-seed average is
   ~1%, concentrated in mixed-size/macro-heavy designs.

RL macro placement does **not** replicate as a QoR lever (verified 3-0):
Google's original gains traced substantially to undisclosed commercial-tool
initialization (~7–10%); SA beats Circuit Training in 6–7 of 9 cases at ~21×
less compute; commercial tools beat CT by 34% routed WL.

## Verified findings (vote = confirm-refute from 3-skeptic panels)

| # | Finding | Vote | Sources |
|---|---|---|---|
| 1 | **C3PO (NVIDIA, ASP-DAC 2026)**: differentiable multi-objective placer — timing (TNS/WNS), routability, WL concurrently, exact gradients in custom CUDA kernels *extending DREAMPlace*, MGDA-style gradient balancing. Architecture verified; its QoR numbers could NOT be verified (PDF was an LFS stub) | 3-0 | research.nvidia.com lu2026aspdac |
| 2 | **RoutePlacer (KDD'24)**: differentiable GNN routability term ⇒ ~13–16% routing-overflow reduction vs DREAMPlace on ISPD2011, WL flat — **5-seed mean±std reported** (rare in this field). The 44% headline requires cell-inflation on both sides and costs ~1.5% WL | 3-0 | arXiv 2406.02651 |
| 3 | **Differentiable STA (DAC'22) / Xplace-Timing (ICCAD'24)**: LSE-smoothed DAG timing as a global objective; HPWL ~flat vs net-weighting's ~4.3% degradation ("free timing" is a property of the differentiable formulation). Headline 32.7%/59.1% WNS/TNS REFUTED as single-config, hand-tuned | 3-0 (headline 1-2) | guozz.cn tdpdac-22, Xplace |
| 4 | **Init refinement (arXiv 2511.10073, 50-seed)**: area-hint init ⇒ peak 2.2%, **average ~1%**, 11/12 cases, gains concentrated in mixed-size/macro-heavy — not std-cell suites | 3-0 | arXiv 2511.10073 |
| 5 | **DG-RePlAce (TCAD'24)**: dataflow/datapath constraints ⇒ 7% routed-WL, 34% TNS vs DREAMPlace on ML accelerators; ablation attributes gains to the injected structure, same Nesterov optimizer. Caveat: baseline may be non-timing-driven | 3-0 | arXiv 2404.13049 |
| 6 | **RL macro placement doesn't replicate** (see headline) | 3-0 | arXiv 2306.09633, 2302.11014, CACM, Nature addendum |
| 7 | **GPU throughput ≠ QoR**: Xplace 2-3× faster, ~0.3% HPWL; Xplace-NN ~0.1% | 3-0 | DAC'22 3530485 |

## Field-wide hazard (confirmed)

Nearly every headline >2% claim in placement is **single-seed and/or
benchmark-heterogeneous**. Only RoutePlacer (5 seeds) and 2511.10073
(50 seeds) report variance — both show real but smaller average gains than
their peaks. Our measured-noise-floor + paired-multi-seed protocol is ahead
of field practice; keep it for everything below.

## Ranked levers for OUR context

| Rank | Lever | Verified effect | Effort | Harness fit |
|---|---|---|---|---|
| 1 | **In-loop differentiable routability term** (RoutePlacer-style penalty or RUDY-gradient) | 13–16% routing overflow, WL flat (multi-seed) | M–L: new loss term + ISPD2011/DAC2012 benchmarks + routability eval | Good — same shape as our `timing_loss` hook (extra differentiable objective term); LLM-evolution can search the penalty form/weighting |
| 2 | **Differentiable timing objective** (LSE-smoothed STA in torch) | TNS/WNS gains with ~flat WL (formulation verified, magnitudes contested) | L: torch STA DAG + ICCAD2015 benchmarks; bypasses x86-only HeteroSTA | Good — `timing_loss` hook already plumbed (Exp 4 infrastructure); evolution searches smoothing/weight schedules *of a real objective* this time |
| 3 | **Structure injection** (dataflow/datapath constraints) | 7% WL / 34% TNS but domain-specific (ML accelerators) | M: needs netlist group extraction | Medium — constraint-extraction function is evolvable |
| 4 | **Init refinement** (area-hint style) | ~1% avg honest, macro-heavy only | S–M via existing `init_positions` hook | Good fit, modest ceiling; pair with macro-heavy benchmarks (MMS/ISPD2005), NOT fft_* |
| — | Schedules, RL policies, kernel speedups | <0.5% / non-replicating / ~0.3% | — | **Closed** (our result + verified findings) |

## Open questions (axis gaps the research couldn't evidence)

1. C3PO's actual QoR methodology (get the real PDF; single-seed?).
2. Do routability/timing gains transfer to std-cell ISPD2015, or do we
   switch benchmarks (ISPD2011/DAC2012/ICCAD2015) with the lever?
3. Axis (d) — what 128 GB unified memory uniquely enables (population/batch
   placement, in-loop surrogates, full-chip GPU DP) — **no surviving claim
   addressed this**; needs a dedicated follow-up search.
4. The legalization/detailed-placement QoR gap — uncovered by all 25
   verified claims; genuinely under-studied (possible niche).

## Recommendation

Lever 1 (differentiable routability) is the best effort/evidence/fit
trade: multi-seed-validated effect, an objective-term shape our hook system
already supports, and an evolvable functional form (the penalty function and
its weighting) where LLM search operates on something with real headroom —
unlike schedules. Lever 2 rides the same harness change and reuses Exp 4
plumbing. Both require adopting the benchmarks the effects were proven on.

Sources: see verified-claim table; full provenance and refuted-claim list in
the deep-research transcript (2026-06-05).
