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
    # Clamp overflow to a sane range so extreme inputs can't destabilize gamma.
    ovf = min(1.0, max(0.0, overflow))

    # Overflow-driven base (DREAMPlace family): smooth gradients while cells
    # are still spread out, sharpening as the layout legalizes.
    base = 4.0 * 10.0 ** ((ovf - 0.1) * 20.0 / 9.0 - 1.0)

    # Normalized optimization progress in [0, 1].
    prog = 0.0
    if total_iterations > 1:
        prog = iteration / float(total_iterations - 1)
        prog = min(1.0, max(0.0, prog))

    # Continuous legality gate: smoothly ramps from ~0 to ~1 as overflow falls
    # through ~0.10. Replaces the hard 0.15 cutoff so the sharpening turns on
    # gradually instead of snapping, avoiding gradient discontinuities.
    legal = 1.0 / (1.0 + math.exp((ovf - 0.10) * 60.0))

    # Late-stage sharpening: once the placement is mostly legal, bias gamma
    # downward so the WA model tracks true HPWL more closely. The push grows
    # with progress but is gated by legality so it never destabilizes the
    # still-spreading early/mid phases. Slightly stronger than the seed because
    # the smooth gate keeps it safe.
    sharpen = 1.0 - 0.55 * legal * (prog ** 1.3)
    gamma = base * sharpen

    # History-driven adaptation: react to the recent HPWL trajectory with
    # proportional (rather than discrete-step) scaling for smoother control.
    if len(hpwl_history) >= 4:
        h = hpwl_history[-4:]
        recent = h[-1]
        prev = (h[0] + h[1] + h[2]) / 3.0
        if prev > 0.0:
            rel = (prev - recent) / prev
            if rel >= 0.0:
                # Improving or flat: tighten the WL approximation, hardest on a
                # plateau (rel~0) and tapering off as real gains accumulate.
                tighten = 0.80 + 0.18 * min(1.0, rel / 0.01)
                gamma *= max(0.80, min(1.0, tighten))
            else:
                # Worsening / oscillating: restore smoothness proportionally to
                # the size of the regression to recover stability.
                gamma *= min(1.30, 1.0 + min(0.30, (-rel) / 0.005 * 0.15))

    return min(50.0, max(0.01, gamma))