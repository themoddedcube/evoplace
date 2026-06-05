import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base anneal: geometric (log-linear) decay along a cosine warp ----
    # cosine warp keeps gamma high a bit longer early, then drops smoothly.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ------------------------------------------------
    # While the layout is still spread out (high overflow) keep gradients
    # smooth; as bins clear, let gamma fall toward the accurate regime.
    # Blend a multiplicative term (scales the anneal) with an additive
    # overflow floor (prevents collapsing gamma while cells still overlap).
    ov_mult = 0.6 + 1.4 * (ov ** 1.3)
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.6)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_floor

    # --- HPWL-history feedback -------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first = window[0]
            last = window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Diverging HPWL -> gradients too noisy: smooth them out.
            if first > 0.0 and last > first * 1.02:
                gamma *= 1.35
            # Steady improvement -> push toward accuracy.
            elif first > 0.0 and last < first * 0.98:
                gamma *= 0.92
            # Plateau (no meaningful improvement) -> sharpen to escape it,
            # but only mildly so we don't oscillate.
            elif prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # --- end-game ceilings: force accuracy late, scaled by remaining overflow
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp ------------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))