import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-cosine gamma schedule with plateau/divergence control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space cosine annealing as the time-driven backbone.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    time_base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the more reliable physical signal than raw iteration count:
    # when cells are still spread (high overflow) keep gamma high regardless of
    # progress; once density resolves, let gamma collapse for accurate HPWL.
    ov_base = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.85)

    # Blend, shifting trust toward the overflow signal as placement matures.
    w_ov = 0.40 + 0.45 * progress
    gamma = (1.0 - w_ov) * time_base + w_ov * ov_base

    # History-based adaptation: react to plateaus and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Diverging HPWL -> gradients too noisy, smooth them out.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent -> push gamma down a touch for sharper HPWL.
            elif last < first * 0.985:
                gamma *= 0.92

            # Stalled improvement -> ease gamma to escape the plateau,
            # but only meaningfully once density has mostly resolved.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80 if ov < 0.12 else 0.95

    # Late-stage ceilings to guarantee fine-tuning, gated on residual overflow.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))