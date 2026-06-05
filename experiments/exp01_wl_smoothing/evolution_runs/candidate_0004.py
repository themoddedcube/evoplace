"""
EVOLVE_TARGET: gamma_schedule
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
    # Canonical DREAMPlace overflow->gamma mapping: γ ≈ 40 at overflow=1.0
    # (smooth gradients while cells spread) down to ≈ 0.4 at overflow=0.1
    # (accurate WL approaching legalization).
    of = min(1.0, max(0.0, overflow))
    base = 4.0 * 10.0 ** ((of - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Late-stage sharpening: as the layout legalizes (low overflow) AND the run
    # nears completion, bias γ downward so the WA model tracks true HPWL. Gated
    # by a smootherstep legality factor (C2-continuous) instead of a hard knee,
    # so the still-spreading early/mid phases are left untouched and the
    # transition introduces no schedule discontinuity that could kick the solver.
    x = min(1.0, max(0.0, 1.0 - of / 0.18))
    legal = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)  # ~0 until overflow < 0.18
    sharpen = 1.0 - 0.50 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # Near-legal accuracy boost: once overflow is tiny, residual error is
    # dominated by WA-model smoothing rather than density, so a modestly sharper
    # model recovers real wirelength. Kept gentle to avoid gradient blow-up.
    if of < 0.05:
        gamma *= 0.92 + 1.6 * of  # 0.92 at of=0 -> 1.0 at of=0.05

    # History-driven adaptation: respond to the recent HPWL trajectory, with the
    # corrective strength scaled to severity so oscillations are damped smoothly
    # rather than via fixed jumps.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev  # >0 improving, <0 worsening
            if -0.001 < rel < 0.001:
                # Plateau: tighten the WL approximation to squeeze out HPWL.
                gamma *= 0.82
            elif rel < 0.0:
                # Worsening / oscillating: restore smoothness in proportion to
                # the regression, capped, to recover stability without overshoot.
                gamma *= min(1.25, 1.0 - 6.0 * rel)

    return min(50.0, max(0.01, gamma))