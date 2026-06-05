import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-gated cosine-exponential gamma anneal with plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth cosine-shaped exponential decay in log-space: stays high while
    # cells are still spreading, then accelerates toward gamma_low near the end.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the physical signal of how clustered the layout still is.
    # While overflow is high we MUST keep gamma high for smooth gradients;
    # as bins clear we trust the schedule's decay and let gamma fall.
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # Blend the time-based decay with the overflow-driven floor, weighting
    # toward overflow early (placement-dominated) and toward time late (refine).
    w_ov = 0.5 * (1.0 - progress) + 0.15
    gamma = (1.0 - w_ov) * base + w_ov * max(base, ov_floor)

    # Trend / plateau control from recent HPWL.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Converged plateau: sharpen the approximation to chase real HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing): gradients too noisy -> smooth more.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            # Healthy steady improvement: gently sharpen.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Late-stage ceilings force fine-tuning unless layout is still congested.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))