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

    # Overflow-adaptive: high overflow -> high gamma (smooth gradients for cell spreading)
    # Maps overflow [0.05, 0.70] -> [0, 1] with sub-linear power curve
    ov_norm = max(0.0, min(1.0, (overflow - 0.05) / 0.65))
    overflow_gamma = gamma_min + (gamma_max - gamma_min) * (ov_norm ** 0.75)

    # Exponential time decay: guarantees monotone reduction even if overflow plateaus
    time_gamma = gamma_min + (gamma_max - gamma_min) * math.exp(-2.8 * t)

    # Blend shifts from overflow-driven (early) to time-driven (late)
    # so a stuck overflow doesn't prevent gamma from falling
    time_weight = 0.3 + 0.5 * t
    gamma = (1.0 - time_weight) * overflow_gamma + time_weight * time_gamma

    # Stagnation detection: if HPWL barely moves in middle/late run, push gamma lower
    # to sharpen the WA-WL approximation and escape flat regions
    if len(hpwl_history) >= 8 and t > 0.4:
        window = hpwl_history[-8:]
        if window[0] > 0:
            improvement = (window[0] - window[-1]) / window[0]
            if improvement < 0.003:
                gamma = max(gamma_min, gamma * 0.85)

    return max(gamma_min, min(gamma_max, gamma))