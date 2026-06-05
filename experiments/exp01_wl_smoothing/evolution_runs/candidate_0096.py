import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware exponential gamma anneal with stagnation control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth-start, fast-finish anneal in log-space (cosine eased).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow blend: while cells are still spread out (high overflow) keep
    # gamma smooth; as the layout legalizes, trust the schedule's low gamma.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    blend = min(1.0, max(0.0, 0.5 + 0.5 * progress))  # late iters weight schedule
    gamma = (1.0 - blend) * ov_target + blend * base

    # Bias up a touch when overflow is high regardless of phase (avoid collapse).
    gamma *= 0.85 + 0.45 * (ov ** 1.2)

    # Trend-based correction from HPWL history.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnating improvement -> push gamma down to sharpen HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing) -> smooth gradients back up.
            if window[0] > 0 and window[-1] > window[0] * 1.02:
                gamma *= 1.40
            elif window[0] > 0 and window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Hard late-phase ceilings so the final layout is HPWL-accurate,
    # but stay smoother if the layout is not yet legal (overflow high).
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))