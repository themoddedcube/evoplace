import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive cosine-annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Geometric (log-space) cosine anneal from high -> low gamma.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the dominant physical signal: while bins remain congested we
    # keep gamma high for smooth spreading gradients; once overflow collapses we
    # let gamma drop toward the accurate-HPWL regime regardless of iteration.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)

    # Blend schedule-time base with overflow target; weight overflow more as we
    # progress, since late-stage accuracy hinges on actual congestion, not time.
    w_ov = 0.35 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # Mild multiplicative coupling so a still-congested late stage stays smooth.
    gamma *= 0.7 + 0.6 * (ov ** 1.2)

    # HPWL-trend feedback: adapt to the optimizer's actual behavior.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Plateau: little improvement -> sharpen toward accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing) -> back off to smoother gradients.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent -> nudge sharper to lock in accuracy.
            elif last < first * 0.985:
                gamma *= 0.93

    # Late-stage ceilings, gated by remaining congestion.
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.1)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.8)

    # Floor early gamma so spreading never collapses while still congested.
    if progress < 0.30 and ov > 0.50:
        gamma = max(gamma, 3.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))