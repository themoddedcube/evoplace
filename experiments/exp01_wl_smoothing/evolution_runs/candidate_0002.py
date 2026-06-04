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

    # Exponential time decay: stays high early, drops sharply near end
    gamma_time = gamma_min + (gamma_max - gamma_min) * math.exp(-5.0 * t)

    # Overflow-adaptive: power-law maps overflow → gamma
    # alpha=1.5 keeps gamma high until overflow is well below 0.5,
    # then drops fast as cells approach legality
    overflow_norm = max(0.0, min(1.0, overflow))
    gamma_overflow = gamma_min + (gamma_max - gamma_min) * (overflow_norm ** 1.5)

    # Conservative blend: keep whichever is higher so we never sharpen prematurely
    gamma = max(gamma_time, gamma_overflow)

    # HPWL stagnation: if HPWL barely moves over last 8 iters, reduce gamma
    # to sharpen gradients and break out of plateau
    if len(hpwl_history) >= 8:
        w_now = hpwl_history[-1]
        w_old = hpwl_history[-8]
        if w_old > 0 and (w_old - w_now) / w_old < 0.002:
            gamma *= 0.85

    return max(gamma_min, min(gamma_max, gamma))