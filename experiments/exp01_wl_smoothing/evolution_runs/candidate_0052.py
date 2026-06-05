import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive cosine-decayed gamma schedule for WA-WL placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Cosine annealing in log-space: smooth high->low, gentle at both ends.
    cos_t = 0.5 * (1.0 + math.cos(math.pi * progress))
    log_gamma = math.log(gamma_low) + (math.log(gamma_high) - math.log(gamma_low)) * cos_t
    base = math.exp(log_gamma)

    # Overflow coupling: when cells are still spread out (high overflow) keep
    # gamma high for smooth gradients; ease down as the layout legalizes.
    overflow_factor = 0.5 + 1.8 * (ov ** 1.2)
    gamma = base * overflow_factor

    # HPWL-history feedback: react to stagnation / divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            # Stagnating: sharpen approximation to chase real HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.65
            # Diverging: smooth gradients back out to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # Late-stage sharpening for accurate final HPWL.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 2.0)

    if gamma != gamma:  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))