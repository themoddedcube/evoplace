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

    # Overflow-adaptive: overflow ~1.0 (clustered) → gamma_max, overflow ~0.1 (spread) → gamma_min
    overflow_norm = max(0.0, min(1.0, (overflow - 0.1) / 0.9))
    # Power < 1 keeps gamma elevated longer during the early dense-packing phase
    gamma_overflow = gamma_min + (gamma_max - gamma_min) * (overflow_norm ** 0.65)

    # Exponential time decay guarantees gamma reaches gamma_min by the end
    gamma_time = gamma_min + (gamma_max - gamma_min) * math.exp(-3.0 * t)

    # Blend: overflow signal dominates early, time decay provides a floor
    gamma = 0.65 * gamma_overflow + 0.35 * gamma_time

    # Stagnation detection: if HPWL barely improving while overflow is still high,
    # nudge gamma up to escape poor local approximations
    if len(hpwl_history) >= 5 and overflow > 0.2:
        recent = hpwl_history[-5:]
        hpwl_change = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-10)
        if hpwl_change > -0.005:  # <0.5% improvement in last 5 iters
            gamma = min(gamma * 1.15, gamma_max)

    return max(gamma_min, min(gamma_max, gamma))