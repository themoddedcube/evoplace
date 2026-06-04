import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    gamma_max = 8.0
    gamma_min = 0.5

    t = iteration / max(total_iterations - 1, 1)
    ov = max(0.01, min(1.0, overflow))

    # Overflow-adaptive: log-space interpolation
    # Gives finer resolution at low overflow (near convergence)
    log_range = math.log(gamma_max) - math.log(gamma_min)
    gamma_overflow = math.exp(math.log(gamma_min) + ov * log_range)

    # Cosine annealing: smooth time-based baseline
    gamma_cosine = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Blend: shift weight from time-based → overflow-adaptive as placement matures
    w_ov = 0.3 + 0.7 * t
    gamma = (1.0 - w_ov) * gamma_cosine + w_ov * gamma_overflow

    # Stagnation escape: if HPWL barely moving but overflow still high, nudge gamma up
    if len(hpwl_history) >= 5 and hpwl_history[-1] > 0:
        window = hpwl_history[-5:]
        variation = (max(window) - min(window)) / hpwl_history[-1]
        if variation < 0.005 and ov > 0.15:
            gamma = min(gamma * 1.12, gamma_max)

    return max(gamma_min, min(gamma_max, gamma))