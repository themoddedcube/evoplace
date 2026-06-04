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
    ov = max(0.0, min(1.0, overflow))

    # Exponential time decay (same rate as baseline)
    gamma_time = gamma_min + (gamma_max - gamma_min) * math.exp(-3.5 * t)

    # Overflow-adaptive with exponent 0.6: stays elevated at medium overflow,
    # drops sharper once density clears
    gamma_ov = gamma_min + (gamma_max - gamma_min) * (ov ** 0.6)

    # Blend max() → min() as overflow clears.
    # High overflow: take max (preserve gamma for exploration, cells still dense)
    # Low overflow: take min (sharpen WA-WL for accuracy, cells spreading out)
    # Transition from max to min as ov falls from 1.0 → 0.2
    w = min(1.0, max(0.0, (1.0 - ov) / 0.8))
    gamma = (1.0 - w) * max(gamma_time, gamma_ov) + w * min(gamma_time, gamma_ov)

    # Context-aware plateau response
    if len(hpwl_history) >= 6:
        window = hpwl_history[-6:]
        if window[0] > 0:
            rel_change = abs(window[-1] - window[0]) / window[0]
            if rel_change < 0.005:
                if ov > 0.3:
                    # Dense + stagnating: boost gamma to smooth landscape and escape
                    gamma = min(gamma * 1.1, gamma_max)
                elif ov < 0.15:
                    # Near-converged + stagnating: reduce for fine-tuning accuracy
                    gamma *= 0.85

    return max(0.01, min(20.0, gamma))