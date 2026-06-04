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

    # Overflow-adaptive: stay high while bins are still dense
    ov = max(0.0, min(1.0, overflow))
    overflow_gamma = gamma_min + (gamma_max - gamma_min) * (ov ** 0.5)

    # Cosine annealing: smooth time-based convergence guarantee
    cosine_gamma = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Exponentially shift weight: overflow-driven early → time-driven late
    w = math.exp(-4.0 * t)
    gamma = w * overflow_gamma + (1.0 - w) * cosine_gamma

    # Plateau escape: if HPWL barely improving, nudge gamma up to escape local minimum
    if len(hpwl_history) >= 10 and t > 0.2:
        recent_mean = sum(hpwl_history[-5:]) / 5.0
        prior_mean = sum(hpwl_history[-10:-5]) / 5.0
        if prior_mean > 0.0 and (prior_mean - recent_mean) / prior_mean < 0.001:
            gamma = min(20.0, gamma * 1.25)

    return max(gamma_min, min(20.0, gamma))