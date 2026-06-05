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
    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes. Maps
    # overflow=1.0 -> ~40, overflow=0.1 -> ~0.4.
    base = 4.0 * 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Legalization state: a SMOOTH ramp as overflow falls, rather than a hard
    # gate. This is the dominant driver of sharpening — gradient noise is a
    # function of how spread-out the cells still are, not of the iteration
    # count. Widened to 0.22 so sharpening engages a touch earlier and more
    # continuously through the mid-to-late transition.
    legal = min(1.0, max(0.0, 1.0 - overflow / 0.22))
    legal = legal * legal * (3.0 - 2.0 * legal)  # smoothstep for gentle onset

    # Late-stage sharpening: bias gamma downward so the WA model tracks true
    # HPWL closely once the placement is mostly legal. Mostly legalization-
    # driven, with a mild progress term so the final iterations push hardest.
    # Stronger ceiling (0.62) than the seed since accuracy near convergence is
    # where HPWL is actually won, but kept bounded to protect stability.
    sharpen = 1.0 - 0.62 * legal * (0.45 + 0.55 * prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0015 < rel < 0.0015:
                # Plateau: tighten the WL approximation to squeeze out HPWL.
                gamma *= 0.80
            elif rel < -0.006:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.20
            elif rel > 0.02 and legal > 0.5:
                # Improving steadily while mostly legal: lean a little further
                # toward accuracy to capitalize on the descent.
                gamma *= 0.93

    return min(50.0, max(0.01, gamma))