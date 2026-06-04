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
    overflow_clamped = max(0.0, min(1.0, overflow))

    # Overflow-adaptive: high overflow → cells still clustering → high gamma
    # Power < 1 keeps gamma elevated longer while overflow is dropping
    gamma_overflow = gamma_min + (gamma_max - gamma_min) * (overflow_clamped ** 0.75)

    # Exponential time decay: guarantees convergence toward low gamma
    # even if overflow plateaus (e.g. congested designs)
    gamma_time = gamma_min + (gamma_max - gamma_min) * math.exp(-3.5 * t)

    # Be conservative: take whichever signal pushes gamma lower
    gamma = min(gamma_overflow, gamma_time)

    # HPWL stagnation: if the last N steps show < threshold improvement,
    # nudge gamma down to sharpen gradients and escape the plateau
    if len(hpwl_history) >= 6:
        window = hpwl_history[-6:]
        if window[0] > 0:
            relative_change = abs(window[-1] - window[0]) / window[0]
            if relative_change < 0.005:
                gamma *= 0.85

    return max(0.01, min(20.0, gamma))