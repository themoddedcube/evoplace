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
    ovf = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace/RePlAce family): smooth gradients while
    # cells are still spread out, sharpening as the layout legalizes.
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Legalization indicator. The placer converges to overflow ~= 0.082, so the
    # gate is widened to 0.20 (the seed's 0.15 left ~half the available sharpening
    # on the table at the real plateau) while staying ~0 during early spreading.
    legal = min(1.0, max(0.0, 1.0 - ovf / 0.20))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. A smooth cosine
    # ramp grows the push with progress without abrupt schedule jumps, and the
    # legal gate keeps the still-spreading early/mid phases smooth and stable.
    ramp = 0.5 - 0.5 * math.cos(math.pi * prog)
    sharpen = 1.0 - 0.52 * legal * ramp
    gamma = base * sharpen

    # History-driven adaptation: bounded, legalization-scaled multipliers instead
    # of discrete jumps, so the response is continuous and noise-resistant.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev  # > 0 improving, < 0 worsening
            if rel < -0.003:
                # Worsening / oscillating: restore smoothness to recover.
                gamma *= 1.0 + 0.18 * min(1.0, (-rel) / 0.01)
            elif rel < 0.0015:
                # Plateau (or barely improving): tighten the WL approximation to
                # squeeze out residual HPWL, harder once the layout is legal.
                gamma *= 1.0 - 0.18 * (0.35 + 0.65 * legal)
            # Still improving strongly: leave smoothness untouched and let it run.

    return min(50.0, max(0.01, gamma))