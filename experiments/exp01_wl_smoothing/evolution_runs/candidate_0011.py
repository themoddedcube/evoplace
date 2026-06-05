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

    # Legality gate: ~0 while the layout is still spreading, ramping smoothly to
    # 1 as overflow drops below ~0.12. Squared so the sharpening stays inert
    # until legalization is genuinely underway, avoiding early destabilization.
    legal = min(1.0, max(0.0, 1.0 - overflow / 0.12))
    legal = legal * legal

    # Late-stage sharpening: once the placement is both legal and late, bias
    # gamma further downward so the WA model tracks true HPWL more closely. A
    # deeper reduction (up to 0.6) than the seed, but tightly gated by `legal`
    # so the still-spreading early/mid phases keep smooth gradients.
    sharpen = 1.0 - 0.60 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory using a
    # smoothed estimate so single-iteration noise does not flip the decision.
    if len(hpwl_history) >= 5:
        h = hpwl_history[-5:]
        recent = (h[-1] + h[-2]) / 2.0
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if -0.0008 < rel < 0.0008:
                # Plateau: progress has stalled. Tighten the WL approximation
                # hard to squeeze out residual HPWL, more aggressively when the
                # layout is already legal (where low gamma is safe).
                gamma *= 0.80 - 0.10 * legal
            elif rel < -0.004:
                # Worsening / oscillating: restore smoothness to recover, with a
                # stronger correction the more legal (and thus noise-prone) we are.
                gamma *= 1.15 + 0.10 * legal
            elif rel > 0.02:
                # Healthy, fast descent: nudge gamma down slightly to keep the
                # model honest while the optimizer is making real progress.
                gamma *= 0.97

    return min(50.0, max(0.01, gamma))