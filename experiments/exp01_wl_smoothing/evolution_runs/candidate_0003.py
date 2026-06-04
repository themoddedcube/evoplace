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
    overflow = max(0.0, min(1.0, overflow))

    # Overflow-exponential: gamma tracks placement quality directly
    # sqrt exponent decays faster early (high overflow), slower late (low overflow)
    gamma_of = gamma_min + (gamma_max - gamma_min) * (overflow ** 0.6)

    # Time-based exponential fallback for early iterations when overflow is unreliable
    gamma_t = max(gamma_min, gamma_max * math.exp(-2.5 * t))

    # Blend: ramp from time-based to overflow-adaptive over first 20% of run
    alpha = min(1.0, t / 0.2)
    gamma = (1 - alpha) * gamma_t + alpha * gamma_of

    # HPWL trend correction: boost gamma if HPWL is worsening, reduce if converging fast
    if len(hpwl_history) >= 5:
        window = hpwl_history[-5:]
        trend = (window[-1] - window[0]) / max(abs(window[0]), 1e-10)
        if trend > 0.02:
            gamma = min(gamma_max, gamma * 1.1)
        elif trend < -0.05:
            gamma = max(gamma_min, gamma * 0.95)

    return max(gamma_min, min(gamma_max, gamma))