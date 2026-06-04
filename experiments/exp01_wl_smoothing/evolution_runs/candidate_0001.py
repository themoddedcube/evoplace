""" ... """

import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """ ... """
    
    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations - 1, 1)
    ov = max(0.0, min(1.0, overflow))

    # Overflow-adaptive: sqrt scaling keeps gamma high while cells are
    # still spreading, then drops sharply once overflow gets small
    ov_driven = gamma_min + (gamma_max - gamma_min) * math.sqrt(ov)

    # Cosine annealing as a time-based anchor (smoother than linear)
    cosine = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Progressive blend: early iterations follow cosine (overflow near 1,
    # not yet informative); later iterations follow overflow signal
    blend = min(1.0, 2.5 * t)
    gamma = (1.0 - blend) * cosine + blend * ov_driven

    # Plateau detection: if HPWL hasn't improved in recent iterations,
    # reduce gamma aggressively to sharpen the WL approximation and escape
    if len(hpwl_history) >= 8:
        recent = hpwl_history[-8:]
        h_max = max(recent)
        if h_max > 0 and (h_max - min(recent)) / h_max < 0.002:
            gamma = max(gamma_min, gamma * 0.65)

    return max(gamma_min, min(gamma_max, gamma))