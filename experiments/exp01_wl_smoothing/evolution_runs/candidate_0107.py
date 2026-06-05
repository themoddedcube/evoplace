import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored cosine gamma decay with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth (cosine-eased) geometric interpolation in log-space from
    # high -> low gamma as placement progresses.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the dominant signal: while bins are congested we keep
    # gradients smooth (high gamma); as the layout legalizes we trust the
    # progress-based decay. Blend a multiplicative and an additive term so
    # neither alone can drive gamma to an extreme.
    ov_mult = 0.6 + 1.4 * (ov ** 1.2)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.5 * base * ov_mult + 0.5 * ov_add

    # History-driven fine adjustment: react to convergence/divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stalled improvement: sharpen toward a more accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging (HPWL climbing): smooth gradients back out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.3
            # Healthy descent: nudge sharper for accuracy.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Late-stage ceilings force accurate (low) gamma for final HPWL,
    # but relax if the layout is still overflowing.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # Guard against the late-stage path collapsing gamma too far, which
    # destabilizes gradients and can blow up HPWL.
    floor = max(gamma_low, 1.5 * (1.0 - progress)) if ov > 0.20 else gamma_low
    gamma = max(gamma, floor)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))