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
    # Clamp overflow defensively; the schedule below assumes overflow in [0, 1].
    ovf = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes. The exponent is
    # tied directly to overflow so gamma decays continuously toward legalization.
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. The push grows
    # with progress but is gated by low overflow so it never destabilizes the
    # still-spreading early/mid phases. Slightly stronger and engaging a touch
    # earlier (overflow < 0.18) than the seed, which empirically squeezed HPWL.
    legal = min(1.0, max(0.0, 1.0 - ovf / 0.18))
    sharpen = 1.0 - 0.52 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory using a
    # longer window for a less noisy slope estimate.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0008 < rel < 0.0008:
                # Plateau: tighten the WL approximation to squeeze out HPWL.
                # A firmer pull (0.80) than the seed, scaled by how legal we are
                # so we never over-sharpen while cells are still overlapping.
                gamma *= 1.0 - 0.20 * legal
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.18
            elif rel > 0.02:
                # Strong, healthy descent: nudge slightly sharper to capitalize.
                gamma *= 0.96

    return min(50.0, max(0.01, gamma))