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

    # Overflow-adaptive: primary signal — overflow directly reflects placement state.
    # Power mapping (exponent < 1) makes gamma more responsive at low overflow,
    # where accuracy matters most.
    ov = max(0.0, min(1.0, overflow))
    gamma_ov = gamma_min + (gamma_max - gamma_min) * (ov ** 0.65)

    # Cosine annealing: time-based fallback — better tail behavior than linear.
    gamma_cos = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Blend: overflow drives early (when t≈0, alpha≈1),
    # cosine takes over late (when t≈1, alpha≈0.05).
    alpha = math.exp(-3.0 * t)
    gamma = alpha * gamma_ov + (1.0 - alpha) * gamma_cos

    # HPWL stagnation: if recent improvement < 0.2%, sharpen gradients by
    # pushing gamma down — avoids getting trapped in a smooth but inaccurate basin.
    if len(hpwl_history) >= 6:
        avg_recent = sum(hpwl_history[-3:]) / 3
        avg_older = sum(hpwl_history[-6:-3]) / 3
        if avg_older > 1e-9 and (avg_older - avg_recent) / avg_older < 0.002:
            gamma = max(gamma_min, gamma * 0.88)

    return max(gamma_min, min(gamma_max, gamma))