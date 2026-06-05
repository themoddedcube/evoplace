import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing for DREAMPlace WA-WL.

    High gamma while cells are still spread (high overflow / early progress)
    for smooth, well-conditioned gradients; smoothly anneals to low gamma for
    accurate HPWL during fine-tuning. Mild plateau/divergence guards nudge the
    schedule without large discontinuous jumps that can destabilize placement.
    """

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

    # --- base log-space cosine anneal (high -> low) ----------------------
    # cos_prog: 0 at start, 1 at end; geometric interpolation in log space
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ----------------------------------------------
    # When overflow is high the placement is still global: bias gamma up so
    # gradients stay smooth. As overflow collapses, release toward `base`.
    # Blend a multiplicative term (relative) with an additive floor (absolute)
    # so neither dominates at the extremes.
    ov_mult = 0.60 + 1.40 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_add

    # --- HPWL trend adaptation ------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: best HPWL barely improving -> sharpen toward accurate
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> soften gradients to recover stability
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Healthy descent: trim gamma slightly for sharper objective
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # --- late-stage accuracy ceilings -----------------------------------
    # Force low gamma late so the reported HPWL reflects the true objective,
    # but relax the ceiling if density has not yet legalized (ov still high).
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))