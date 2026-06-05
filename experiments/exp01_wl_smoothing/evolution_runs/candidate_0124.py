import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored cosine gamma schedule with gentle plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth cosine geometric interpolation from high -> low over the run.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow-driven anchor: while cells are still spread (high overflow) we
    # want larger gamma for smooth, well-conditioned gradients; as the layout
    # legalizes (overflow -> 0) we let gamma fall toward the accurate regime.
    ov_anchor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)

    # Blend the schedule-driven and overflow-driven targets. Early on, trust the
    # schedule; later, trust the physical overflow signal more.
    w_ov = 0.30 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_anchor

    # Plateau / divergence response from HPWL trend.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stalled improvement: sharpen slightly to refine wirelength.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # HPWL climbing: gradients likely too noisy -> smooth them out.
            if window[0] > 0 and window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Healthy descent: nudge toward accuracy.
            elif window[0] > 0 and window[-1] < window[0] * 0.98:
                gamma *= 0.96

    # Late-stage accuracy ceilings, relaxed when density is still poor.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))