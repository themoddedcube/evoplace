import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule with plateau adaptation."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base anneal: cosine in log-space for a smooth high->low glide ---
    # cosine factor goes 1 -> 0 over progress, shaped so most of the
    # smoothing happens early and fine-tuning dominates the tail.
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    log_high, log_low = math.log(gamma_high), math.log(gamma_low)
    base = math.exp(log_low + (log_high - log_low) * cos_factor)

    # --- overflow coupling: stay smooth while cells are still congested ---
    # When overflow is high (cells not yet spread), bias gamma up; once
    # density resolves, let the anneal pull gamma down for accurate HPWL.
    overflow_factor = 0.55 + 1.9 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- history-driven adaptation ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # plateau: progress has stalled -> sharpen (lower gamma) to refine
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # divergence: HPWL trending up -> smooth (raise gamma) to stabilize
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # --- tail clamp: force accurate regime near the end ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.2)

    # --- final safety clamp ---
    if gamma != gamma:  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))