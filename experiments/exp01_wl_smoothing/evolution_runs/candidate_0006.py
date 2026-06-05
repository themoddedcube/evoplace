import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """
    # progress in [0, 1]
    t = iteration / max(1, total_iterations)

    # Overflow is the primary driver in DREAMPlace-style placement:
    # while cells are still overlapping (high overflow) we want large gamma
    # for smooth gradients; as the layout legalizes (overflow -> 0) we drop
    # gamma to sharpen the HPWL approximation.
    of = min(1.0, max(0.0, overflow))

    # Base exponential map from overflow: ~8.0 at high overflow,
    # ~0.5 near zero overflow. 10^(2*(of-0.2)) gives a smooth sweep.
    gamma_of = 0.5 * 10.0 ** (2.0 * of)

    # Cosine-annealed progress floor so we still cool down even if overflow
    # stalls on a plateau. Goes from 1.0 (start) to ~0.0 (end).
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    gamma_progress = 0.5 + 7.5 * cos_factor

    # Blend: overflow dominates early, progress takes over as it shrinks.
    w = of  # weight toward overflow signal when cells still overlap
    gamma = w * gamma_of + (1.0 - w) * gamma_progress

    # Plateau detection: if recent HPWL has stopped improving, nudge gamma
    # down to refine the approximation and escape the flat region.
    if len(hpwl_history) >= 4:
        recent = hpwl_history[-4:]
        spread = max(recent) - min(recent)
        denom = abs(recent[-1]) + 1e-12
        if spread / denom < 1e-3:
            gamma *= 0.7

    return min(50.0, max(0.01, gamma))