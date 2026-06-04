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

    # Overflow-adaptive: overflow ~0.8 at start, ~0.05 at convergence
    ov = max(0.05, min(0.80, overflow))
    ov_frac = (ov - 0.05) / 0.75          # normalize to [0, 1]
    gamma_ov = gamma_min + (gamma_max - gamma_min) * (ov_frac ** 0.65)

    # Exponential time decay: monotone fallback if overflow stalls
    gamma_t = gamma_min + (gamma_max - gamma_min) * math.exp(-3.5 * t)

    # Blend: overflow signal dominates early, time signal dominates late
    w = math.exp(-2.5 * t)
    gamma = w * gamma_ov + (1.0 - w) * gamma_t

    # Mid-phase stagnation: if HPWL barely moves, nudge gamma down to escape plateau
    if len(hpwl_history) >= 10 and 0.2 < t < 0.7:
        recent = hpwl_history[-5:]
        mean_h = sum(recent) / len(recent)
        if mean_h > 0 and abs(recent[-1] - recent[0]) / mean_h < 0.003:
            gamma *= 0.85

    return max(gamma_min, min(20.0, gamma))