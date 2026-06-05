import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space cosine descent: high gamma early, low gamma late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow drives gamma: spread-out cells (high overflow) want smoother
    # gradients; near-legal layouts (low overflow) want accurate HPWL.
    ov_factor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)

    # Blend schedule-driven and overflow-driven targets. Weight shifts toward
    # the overflow signal late in the run when geometry matters most.
    w = 0.35 + 0.30 * progress
    gamma = (1.0 - w) * base + w * ov_factor

    # Plateau / divergence adaptation from recent HPWL trend.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Stalled improvement: sharpen to chase finer HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging: smooth gradients to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.95

    # Late-stage ceilings keep approximation accurate once cells are placed,
    # but stay permissive while density is still high.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))