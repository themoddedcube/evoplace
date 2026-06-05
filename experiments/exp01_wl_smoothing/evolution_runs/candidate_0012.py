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
    ovf = min(1.0, max(0.0, overflow))
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: as the placement legalizes, bias gamma downward so
    # the WA model tracks true HPWL more closely. Use a smooth, continuous gate
    # in overflow (no hard threshold) so the transition never destabilizes the
    # still-spreading phases, and let it deepen with optimization progress.
    # legal -> 1 as overflow -> 0, with most of the action below ovf ~ 0.15.
    legal = math.exp(-ovf / 0.07)              # smooth: ~1 near legal, decays fast
    sharpen = 1.0 - 0.55 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react proportionally to the recent HPWL trend
    # instead of with coarse thresholds. A clean, steady descent lets us tighten
    # the approximation; stalls/oscillation call for more smoothness.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev  # >0 improving, <0 worsening
            # Variance of the recent window: high variance => noisy gradients.
            mean = sum(h) / len(h)
            if mean > 0.0:
                var = sum((x - mean) ** 2 for x in h) / len(h)
                noise = math.sqrt(var) / mean
            else:
                noise = 0.0

            if -0.0015 < rel < 0.0015:
                # Plateau: tighten WL approximation to squeeze out HPWL, but
                # tighten less when the window is noisy (a noisy plateau is
                # really oscillation in disguise).
                tighten = 0.18 * (1.0 - min(1.0, noise / 0.01))
                gamma *= (1.0 - tighten)
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover,
                # scaled by how badly it is regressing.
                boost = min(0.30, 0.15 + 30.0 * (-rel - 0.004))
                gamma *= (1.0 + boost)
            elif noise > 0.02:
                # Improving but jittery: a touch more smoothing stabilizes it.
                gamma *= 1.05

    return min(50.0, max(0.01, gamma))