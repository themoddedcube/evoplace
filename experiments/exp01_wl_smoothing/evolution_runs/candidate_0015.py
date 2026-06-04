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

    # Overflow-adaptive component: map placement state → gamma
    # High overflow = cells still spreading = need smooth gradients
    ov_high = 0.9
    ov_low = 0.05
    ov_norm = (overflow - ov_low) / max(ov_high - ov_low, 1e-6)
    ov_norm = max(0.0, min(1.0, ov_norm))
    # Sub-linear mapping: faster drop in mid-overflow range
    gamma_ov = gamma_min + (gamma_max - gamma_min) * (ov_norm ** 0.8)

    # Iteration-based exponential decay: ensures monotonic progress
    # Fast initial drop, then slow — better than linear for early exploration
    gamma_t = gamma_min + (gamma_max - gamma_min) * math.exp(-5.0 * t)

    # Blend: first third of run leans on iteration signal (overflow may be flat),
    # then hands off to overflow signal once cells have started spreading
    w = min(1.0, t * 3.0)
    gamma = (1.0 - w) * gamma_t + w * gamma_ov

    # Plateau detection: if HPWL hasn't improved over last 10 iters,
    # nudge gamma up slightly to smooth gradients and escape the basin
    if len(hpwl_history) >= 10 and t > 0.2:
        recent = sum(hpwl_history[-5:]) / 5
        older  = sum(hpwl_history[-10:-5]) / 5
        if older > 0 and (older - recent) / older < 5e-4:
            gamma = min(gamma * 1.15, gamma_max * 0.7)

    return max(gamma_min, min(gamma_max, gamma))