"""
EVOLVE_TARGET: gamma_schedule

Initial seed program for OpenEvolve (Experiment 1: WL Smoothing Schedule).

This is the EDITABLE function that the LLM evolution engine will mutate.
The evaluation harness (evaluator_wrapper.py) calls this function and
measures the resulting HPWL on the benchmark.

RULES (do not change these comments — the evolution engine reads them):
- Function signature must be preserved exactly
- Only modify the function body
- No new imports allowed
- No external state or file I/O
- Return value must be a float in range [0.01, 50.0]
"""

import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """
    WA-WL smoothness schedule: returns γ for the weighted-average wirelength model.

    γ controls the tradeoff between WL accuracy and gradient smoothness:
    - High γ (~40): smooth gradients, inaccurate HPWL approximation
    - Low γ (~0.4): accurate HPWL, but gradients become noisy near convergence

    Args:
        iteration: current iteration number (0 to total_iterations-1)
        total_iterations: total planned iterations
        overflow: current density overflow (0.0 = no overflow, 1.0 = full overflow)
        hpwl_history: list of HPWL values at previous iterations

    Returns:
        gamma: float in [0.01, 50.0]
    """
    # Clamp overflow defensively (legalization can briefly report >1 or <0).
    ovf = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace/RePlAce family): γ maps exponentially to
    # overflow so gradients stay smooth while cells are spread out and sharpen
    # as the layout legalizes. Slightly steeper slope than the seed concentrates
    # the high-γ regime in the early phase and frees the tail to go sharper.
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: once the placement is mostly legal, bias γ downward
    # so the WA model tracks true HPWL more closely. The push grows with progress
    # but is gated by low overflow so it never destabilizes the spreading phase.
    # A smooth (squared) legalization gate avoids the abrupt onset of the linear
    # version, and a deeper terminal factor lets the very end approximate HPWL.
    legal = min(1.0, max(0.0, 1.0 - ovf / 0.15))  # ~0 until overflow < 0.15
    legal = legal * legal
    sharpen = 1.0 - 0.55 * legal * (prog ** 1.5)
    gamma = base * sharpen

    # Progress-dependent floor: allow the WA model to become genuinely sharp at
    # the very end (where overflow is tiny) without letting γ collapse mid-run.
    floor = 0.05 + 0.35 * (1.0 - prog) + 4.0 * ovf
    gamma = max(gamma, floor * 0.0 + min(gamma, gamma))  # no-op guard; see below
    if gamma < floor and ovf < 0.1:
        gamma = floor

    # History-driven adaptation: react to the recent HPWL trajectory.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0008 < rel < 0.0008:
                # Plateau: tighten the WL approximation to squeeze out HPWL.
                gamma *= 0.82
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.18

    return min(50.0, max(0.01, gamma))