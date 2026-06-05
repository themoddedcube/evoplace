import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored geometric decay with gentle stagnation control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base schedule: smooth geometric (log-linear) decay in time.
    # Cosine easing keeps gamma high a bit longer early, then drops fast.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    time_base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow anchor: physical state matters more than the clock.
    # When the layout is still spread (high overflow), keep gradients smooth;
    # once cells settle (low overflow), trust the sharp HPWL approximation.
    ov_anchor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.3)

    # Blend: lean on overflow early (geometry-driven), on the clock late.
    w_time = 0.35 + 0.30 * progress
    gamma = (1.0 - w_time) * ov_anchor + w_time * time_base

    # --- HPWL feedback: nudge, never multiply wildly.
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)

            # Stagnation: HPWL barely improving -> sharpen to escape plateau.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

            # Divergence: HPWL climbing -> smooth out the noisy gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25
            # Steady descent -> let it ride, ease down a touch.
            elif window[-1] < window[0] * 0.99:
                gamma *= 0.96

    # --- Late-stage ceiling: force accuracy once nearly converged,
    # but stay smoother if density has not yet legalized.
    if progress > 0.88:
        gamma = min(gamma, 1.2 if ov > 0.08 else 0.6)
    elif progress > 0.72:
        gamma = min(gamma, 2.2 if ov > 0.08 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))