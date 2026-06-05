import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven geometric gamma schedule with progress anneal and
    HPWL-trend adaptation for WA-WL smoothing in differentiable placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Primary signal: overflow. Geometric interpolation in log-space so gamma
    # spans the full range smoothly as the placement de-clusters. The exponent
    # makes gamma fall off quickly once overflow drops below ~0.2 (cells are
    # mostly spread), shifting from smooth gradients to accurate HPWL.
    ov_shaped = ov ** 0.85
    gamma_ov = math.exp(log_lo + (log_hi - log_lo) * ov_shaped)

    # Secondary signal: iteration progress as a cosine high->low anneal, so
    # even if overflow plateaus we keep driving toward accuracy late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # Blend in log-space; overflow dominates, progress nudges the anneal.
    w = 0.7
    gamma = math.exp(w * math.log(gamma_ov) + (1.0 - w) * math.log(gamma_prog))

    # HPWL-trend adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            if window[-1] > window[0] * 1.02:        # diverging -> smooth more
                gamma *= 1.25
            elif window[-1] < window[0] * 0.985:     # converging -> sharpen
                gamma *= 0.92
            else:                                    # plateau -> sharpen harder
                gamma *= 0.85

    # Guarantee accurate HPWL once nearly legal.
    if progress > 0.9 and ov < 0.08:
        gamma = min(gamma, 0.6)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))