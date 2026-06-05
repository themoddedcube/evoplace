import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-space gamma schedule with progress floor
    and gentle plateau/divergence adaptation."""

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

    # --- primary driver: overflow ---
    # Overflow is the physical spreading signal in DREAMPlace. When cells are
    # still overlapping (high overflow) we want smooth gradients (high gamma);
    # once spread (low overflow) we sharpen to recover accurate HPWL.
    # Interpolate geometrically (linear in log-space) so the transition is
    # smooth across the whole [low, high] range.
    ov_t = ov ** 0.85                   # mild convexity: hold high gamma a bit longer
    ov_gamma = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_t)

    # --- secondary driver: iteration progress (cosine in log-space) ---
    # Acts as a backstop so gamma decays even if overflow plateaus high.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: overflow dominates early/mid, progress enforces late annealing.
    w = progress ** 1.5                  # shift weight toward progress near the end
    gamma = (1.0 - w) * ov_gamma + w * min(ov_gamma, prog_gamma)

    # --- HPWL feedback: react to plateau / divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Diverging HPWL -> too noisy, smooth out.
            if last > first * 1.02:
                gamma *= 1.30
            # Plateaued (negligible improvement) -> sharpen to escape.
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # Healthy descent -> nudge sharper.
            elif last < first * 0.98:
                gamma *= 0.95

    # --- late-stage ceilings for accurate final HPWL ---
    if progress > 0.85:
        gamma = min(gamma, 1.3 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    # --- final guards ---
    if gamma != gamma:                  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))