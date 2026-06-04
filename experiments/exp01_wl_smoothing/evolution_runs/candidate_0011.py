"""..."""

import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """..."""

    gamma_max = 8.0
    gamma_min = 0.5
    t = iteration / max(total_iterations - 1, 1)
    overflow_clamped = max(0.0, min(1.0, overflow))

    # Overflow-adaptive: sqrt keeps gamma higher while overflow is moderate,
    # then drops sharply as overflow approaches 0 (tight placement phase)
    gamma_overflow = gamma_min + (gamma_max - gamma_min) * math.sqrt(overflow_clamped)

    # Cosine annealing: smooth monotone time decay as secondary signal
    gamma_cosine = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Exponential weight: overflow signal dominates early iterations,
    # time-based cosine takes over as overflow becomes noisy near zero
    w = math.exp(-3.0 * t)
    gamma = w * gamma_overflow + (1.0 - w) * gamma_cosine

    # Accelerate decay when HPWL is converging — reward progress with
    # more accurate (lower gamma) approximation
    if len(hpwl_history) >= 6:
        h_old, h_new = hpwl_history[-6], hpwl_history[-1]
        if h_old > 0 and h_new < h_old * 0.99:
            gamma *= 0.95

    return max(gamma_min, min(20.0, gamma))