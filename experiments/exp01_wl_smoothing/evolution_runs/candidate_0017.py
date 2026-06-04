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

    # Overflow-adaptive: gamma tracks placement legality state directly.
    # overflow=1.0 → gamma_max; overflow=0.05 → gamma_min.
    # Power < 1 makes it responsive to early overflow drops.
    ov = max(0.05, min(1.0, overflow))
    ov_norm = (ov - 0.05) / 0.95
    gamma_ov = gamma_min + (gamma_max - gamma_min) * (ov_norm ** 0.7)

    # Time-based: superlinear exponent keeps gamma high early then drops fast.
    # exp(-3.5 * t^1.5): at t=0.5 ≈ 2.7, at t=0.9 ≈ 0.88 vs linear 4.25 / 2.75.
    gamma_t = gamma_min + (gamma_max - gamma_min) * math.exp(-3.5 * t ** 1.5)

    # Neither driver should artificially keep gamma elevated.
    gamma = min(gamma_ov, gamma_t)

    # Plateau detection: if HPWL stagnates, push gamma lower to sharpen gradients.
    if len(hpwl_history) >= 8:
        recent_min = min(hpwl_history[-4:])
        past_min = min(hpwl_history[-8:-4])
        if past_min > 0 and recent_min > 0:
            rel_improvement = (past_min - recent_min) / past_min
            if rel_improvement < 0.003:
                gamma = max(gamma_min, gamma * 0.80)

    return max(gamma_min, min(gamma_max, gamma))