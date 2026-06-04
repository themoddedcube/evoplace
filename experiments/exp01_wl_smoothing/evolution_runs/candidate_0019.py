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
    overflow_clamped = max(0.01, min(1.0, overflow))

    # Cosine annealing as time-based baseline
    gamma_cosine = gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))

    # Overflow-adaptive: gamma tracks placement spread directly
    # sqrt slows the drop so early spreading doesn't cut gamma too fast
    gamma_adaptive = gamma_min + (gamma_max - gamma_min) * math.sqrt(overflow_clamped)

    # Take the minimum: be more aggressive whenever either signal says to
    gamma = min(gamma_cosine, gamma_adaptive)

    # Fine-tuning clamp: once nearly converged, pin to low gamma for HPWL accuracy
    if overflow_clamped < 0.1 and t > 0.7:
        gamma = min(gamma, gamma_min * 2.0)

    # HPWL stagnation: if improvement has stopped mid-run, lower gamma for sharper signal
    if len(hpwl_history) >= 8 and t > 0.35:
        window = hpwl_history[-8:]
        if window[0] > 0:
            rel_change = (window[0] - window[-1]) / window[0]
            if rel_change < 0.002:
                gamma *= max(0.65, 1.0 - 0.6 * (t - 0.35))

    return max(0.01, min(20.0, gamma))