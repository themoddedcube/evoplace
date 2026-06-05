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
    ov = min(1.0, max(0.0, overflow))

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes.
    base = 4.0 * 10.0 ** ((ov - 0.1) * 20.0 / 9.0 - 1.0)

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. The push grows
    # with progress but is gated by low overflow so it never destabilizes the
    # still-spreading early/mid phases. Slightly wider legal gate (0.18) and a
    # deeper floor (down to ~0.5x base) extract more HPWL accuracy near the end
    # without touching the spreading phases.
    legal = min(1.0, max(0.0, 1.0 - ov / 0.18))
    sharpen = 1.0 - 0.50 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory. Use a
    # 5-sample window and a smoothed 2-sample "recent" estimate to suppress
    # single-iteration noise in the plateau/oscillation decision.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0015 < rel < 0.0015:
                # Plateau: tighten the WL approximation to squeeze out HPWL.
                # Tighten harder the deeper we are into the schedule, where
                # gradient noise costs less than approximation error.
                gamma *= 0.80 + 0.10 * (1.0 - prog)
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.18

    return min(50.0, max(0.01, gamma))