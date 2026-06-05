import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-led, progress-annealed gamma schedule with trend feedback."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: density-coupled smoothing (DREAMPlace style). While cells
    # overlap (high overflow) gamma stays high for smooth, well-behaved
    # gradients; as the layout legalizes (overflow -> 0) gamma decays
    # geometrically toward the accurate low value.
    ov_shaped = ov ** 0.8
    log_span = math.log(gamma_high / gamma_low)
    gamma_ov = gamma_low * math.exp(log_span * ov_shaped)

    # Secondary driver: cosine-annealed progress schedule, used as a descending
    # ceiling so gamma still trends down even if overflow plateaus.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: overflow leads early, progress takes over late. Never exceed the
    # smoothness implied by the larger of the two drivers.
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_prog)
    gamma = min(gamma, max(gamma_ov, gamma_prog))

    # HPWL trend feedback: stabilize divergence, escape plateaus, reward descent.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6]

            if last > first * 1.01:            # diverging -> smooth more
                gamma *= 1.25
            elif prev > 0 and (prev - best_recent) / prev < 5e-4:  # stalled
                gamma *= 0.85
            elif last < first * 0.99:          # healthy descent -> sharpen mildly
                gamma *= 0.97

    # End-game: force accurate HPWL near convergence, but respect leftover overflow.
    if progress > 0.88:
        gamma = min(gamma, 1.2 if ov > 0.08 else 0.6)
    elif progress > 0.72:
        gamma = min(gamma, 2.2 if ov > 0.08 else 1.2)

    # Early floor: keep gradients from getting noisy while cells are still messy.
    if progress < 0.5 and ov > 0.5:
        gamma = max(gamma, 2.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))