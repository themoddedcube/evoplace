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

    # Overflow-adaptive component: primary signal
    # overflow ~1.0 early (clustered) → ~0.05 at convergence
    overflow_norm = max(0.0, min(overflow, 1.0))
    # sqrt mapping: stays high while overflow is high, drops sharply near zero
    overflow_gamma = gamma_min + (gamma_max - gamma_min) * (overflow_norm ** 0.55)

    # Cosine annealing: smooth time-based decay as secondary signal
    cosine_gamma = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Blend: ramp overflow weight in over first 10% of iterations
    alpha = min(1.0, t / 0.1)
    gamma = (1.0 - alpha) * gamma_max + alpha * (0.65 * overflow_gamma + 0.35 * cosine_gamma)

    # HPWL trend: if HPWL is rising (oscillation/escape), soften gamma
    if len(hpwl_history) >= 3:
        h = hpwl_history
        if h[-1] > h[-2] and h[-2] > h[-3]:
            gamma = min(gamma * 1.2, gamma_max)
        elif len(h) >= 2 and h[-2] > 1e-10:
            rel_drop = (h[-2] - h[-1]) / h[-2]
            if rel_drop > 0.03:  # fast convergence → sharpen faster
                gamma = max(gamma * 0.92, gamma_min)

    # Late stage: aggressively push toward gamma_min for accurate HPWL
    if t > 0.8:
        late_t = (t - 0.8) / 0.2
        gamma = gamma * (1.0 - late_t) + gamma_min * late_t

    return max(gamma_min, min(gamma_max, gamma))