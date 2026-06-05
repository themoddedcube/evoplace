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
    # are still spread out, sharpening as the layout legalizes.
    base = 4.0 * 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. The push grows
    # with progress but is gated by low overflow so it never destabilizes the
    # still-spreading early/mid phases. A two-tier gate lets a gentle nudge
    # begin earlier (overflow < 0.25) while the aggressive squeeze waits for
    # near-legal layouts (overflow < 0.10), where accuracy matters most.
    legal = min(1.0, max(0.0, 1.0 - overflow / 0.15))   # ~0 until overflow < 0.15
    near = min(1.0, max(0.0, 1.0 - overflow / 0.10))    # ~0 until overflow < 0.10
    early = min(1.0, max(0.0, 1.0 - overflow / 0.25))   # ~0 until overflow < 0.25
    sharpen = 1.0 - 0.50 * legal * (prog ** 1.5) \
                  - 0.18 * near * prog \
                  - 0.06 * early * (prog ** 2.0)
    sharpen = max(0.30, sharpen)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory using a
    # smoothed estimate of the relative improvement rate.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0005 < rel < 0.0015:
                # Plateau / diminishing returns: tighten the WL approximation
                # harder when overflow is already low (cost of noise is small,
                # benefit of accuracy is high).
                gamma *= 0.80 + 0.05 * overflow
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover, but
                # less abruptly so we don't overshoot back to a coarse model.
                gamma *= 1.12
            elif rel > 0.02:
                # Strong, healthy descent: hold gamma slightly smoother to keep
                # the trajectory stable while progress is fast.
                gamma *= 1.03

    return min(50.0, max(0.01, gamma))