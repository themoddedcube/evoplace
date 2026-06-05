import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive geometric gamma annealing for differentiable placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:            # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: geometric (log-linear) decay on a cosine-warped clock ---
    # Cosine warp holds gamma high a bit longer early (cells still clustering),
    # then accelerates the descent toward the accurate-HPWL regime late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # Overflow is the physical signal of how spread-out the layout still is.
    # While bins are congested we want smoother (higher) gradients; once density
    # relaxes we trust the sharper, more accurate low-gamma approximation.
    # Blend a multiplicative term (scales the schedule) with an additive floor
    # derived purely from overflow so a stuck-high-overflow run never collapses
    # gamma prematurely, and a well-spread run is free to sharpen.
    ov_mult = 0.70 + 0.60 * ov                      # in [0.70, 1.30]
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.65 * (base * ov_mult) + 0.35 * ov_floor

    # --- HPWL feedback (gentle, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0.0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau: best HPWL barely improving -> sharpen to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging: wirelength climbing -> smooth gradients to recover.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> nudge sharper to lock in gains.
            elif last < first * 0.98:
                gamma *= 0.93

    # --- late-stage ceilings for accurate final HPWL ---
    # Only force gamma low once density has actually relaxed; if overflow is
    # still high near the end, keep some smoothing so the layout can settle.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp ---
    if gamma != gamma or gamma == float("inf"):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))