import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement.

    Strategy: keep gamma high while cells are still spread out (high
    overflow / early progress) for smooth, well-conditioned gradients,
    then anneal toward a low gamma for an accurate HPWL approximation
    during fine-tuning. Overflow is the primary driver; iteration
    progress is a secondary safety floor so we never over-smooth late.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- overflow-driven core (primary signal) ---------------------------
    # Map overflow -> [0,1] anneal coordinate. When the layout is still
    # congested (ov high) we sit near gamma_high; as the density relaxes
    # we slide toward gamma_low. A gentle power shapes the transition so
    # most of the descent happens once overflow drops below ~0.5.
    ov_coord = ov ** 0.65
    gamma_ov = math.exp(log_lo + (log_hi - log_lo) * ov_coord)

    # --- progress-driven core (secondary signal) -------------------------
    # Cosine anneal in log-space provides a smooth monotone backbone so
    # the schedule still descends even if overflow plateaus early.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # Blend: lean on overflow early, on progress (the guaranteed
    # descent) late. This keeps gamma high while genuinely congested
    # but forces fine-tuning to actually happen by the end.
    w_ov = 1.0 - 0.5 * progress
    gamma = w_ov * gamma_ov + (1.0 - w_ov) * gamma_prog

    # --- HPWL-trend feedback --------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau: little improvement -> sharpen to refine HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing) -> smooth gradients to recover.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent -> nudge slightly sharper.
            elif last < first * 0.98:
                gamma *= 0.92

    # --- late-stage ceilings (force accurate HPWL near the end) ---------
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:  # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))