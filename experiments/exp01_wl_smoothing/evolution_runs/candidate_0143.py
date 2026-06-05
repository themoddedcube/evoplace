import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with a progress-based cooling floor."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Overflow is the primary physical signal in DREAMPlace-style placement:
    # high overflow -> cells still spreading -> smooth (high) gamma for stable
    # gradients; low overflow -> placement settled -> sharpen (low gamma) for an
    # accurate HPWL approximation. Interpolate geometrically in log-space so the
    # transition is smooth and never overshoots the [gamma_low, gamma_high] band.
    ov_gamma = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.5)

    # Progress gives a guaranteed cooling trajectory so gamma cannot stay high
    # indefinitely if overflow plateaus early.
    prog_cos = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** prog_cos

    # Overflow dominates; progress guarantees monotone-ish cooling.
    gamma = 0.65 * ov_gamma + 0.35 * prog_gamma

    # Conservative HPWL-trend feedback. Aggressive multipliers (large jumps on
    # every up/down tick) tend to oscillate and destabilize placement, so the
    # corrections here are small and only fire on clear, sustained signals.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            first = recent[0]
            last = recent[-1]
            best_recent = min(recent)
            if last > first * 1.01:
                # HPWL rising -> approximation too sharp -> smooth slightly.
                gamma *= 1.15
            elif first > 0 and (first - best_recent) / first < 1e-3:
                # Plateaued with no improvement -> sharpen slightly to fine-tune.
                gamma *= 0.90

    # Late-stage caps for accurate final HPWL, but never starve gradients while
    # density is still poor (overflow high).
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))