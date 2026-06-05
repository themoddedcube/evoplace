import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with cosine base decay and
    plateau/divergence adaptation. High gamma while cells are clustered
    (high overflow), low gamma for HPWL-accurate fine-tuning at the end."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    if ov != ov:  # NaN guard
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Cosine annealing base: smooth, slow-fast-slow descent in log space.
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_g = math.log(gamma_low) + cos * (math.log(gamma_high) - math.log(gamma_low))
    base = math.exp(log_g)

    # Overflow coupling: placement still spread out -> keep gamma smooth.
    # Dominant driver early; tapers as a multiplicative correction.
    overflow_factor = 0.5 + 2.0 * (ov ** 1.3)
    gamma = base * overflow_factor

    # History-based adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: progress stalled -> sharpen to escape flat region.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Divergence: HPWL climbing -> smooth gradients to re-stabilize.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # Late-phase cap enforces accurate HPWL once layout is mostly settled,
    # but relax the cap if density is still high (premature would harm legality).
    if progress > 0.85:
        cap = 1.0 + 3.0 * ov
        gamma = min(gamma, cap)
    if progress > 0.95:
        gamma = min(gamma, 0.8 + 2.0 * ov)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))