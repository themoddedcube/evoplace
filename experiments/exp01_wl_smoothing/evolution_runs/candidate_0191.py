import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored gamma schedule for WA-WL global placement.

    Couples gamma primarily to the live overflow signal (the most reliable
    proxy for how clustered/legal the layout is) with a gentle cosine
    progress backbone, then applies light HPWL-trend trimming and
    late-stage ceilings to lock in an accurate final approximation.
    """

    # --- sanitize inputs ---------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- overflow-anchored core -------------------------------------------
    # Overflow is the dominant driver: when the layout is still spread out
    # (high overflow) we want smooth gradients; once bins relax we sharpen.
    # Geometric interpolation in log-space keeps the transition smooth and
    # always inside [gamma_low, gamma_high].
    ov_term = ov ** 0.85                           # mild concavity -> stay smooth longer
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_term)

    # --- progress backbone (cosine annealing) -----------------------------
    # Provides decay even when overflow plateaus; weighted under the overflow
    # signal so it never forces sharpening before the layout is ready.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: overflow leads, progress assists. As placement matures
    # (low overflow) the progress term is allowed to pull gamma further down.
    w_ov = 0.65
    gamma = w_ov * gamma_ov + (1.0 - w_ov) * gamma_prog

    # --- HPWL-trend adaptation --------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Diverging HPWL -> gradients too noisy, smooth back up.
            if last > first * 1.02:
                gamma *= 1.30
            # Stalled improvement -> sharpen to chase real wirelength.
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # Healthy steady descent -> nudge sharper, gently.
            elif last < first * 0.985:
                gamma *= 0.95

    # --- late-stage accuracy ceilings -------------------------------------
    # In the endgame the WA-WL approximation must be tight, but only if the
    # layout is already nearly legal (low overflow); otherwise keep some
    # smoothing to avoid locking in overlaps.
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)
    elif progress > 0.65:
        gamma = min(gamma, 3.5 if ov > 0.12 else 2.0)

    # --- final clamp -------------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))