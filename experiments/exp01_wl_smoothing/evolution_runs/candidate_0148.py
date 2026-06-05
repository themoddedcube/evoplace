import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven annealed gamma schedule for WA-WL smoothing."""

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

    # --- primary driver: overflow coupling (log-linear, DREAMPlace-style)-
    # gamma should track density directly: while bins are packed (high
    # overflow) gradients must stay smooth; as overflow collapses toward 0
    # the wirelength approximation can be made accurate. Interpolate
    # geometrically in overflow so the high-gamma regime is held until the
    # legalization front actually clears.
    ov_warp = ov ** 0.85                          # hold high gamma a touch longer
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_warp)

    # --- secondary driver: schedule progress -----------------------------
    # Even if overflow lingers, force a slow geometric decay with iteration
    # so the run cannot stall at high gamma forever.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Take the smaller (more accurate) of the two pressures, then blend so
    # neither driver alone can pin gamma artificially high.
    gamma = 0.65 * min(gamma_ov, gamma_prog) + 0.35 * gamma_ov

    # --- bounded adaptive feedback from HPWL trajectory ------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: improvement stalled -> sharpen (lower gamma) to chase
            # a more accurate wirelength gradient.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth gently to re-stabilize.
            # Bounded (<=1.15x) so feedback can never amplify a blow-up.
            if window[-1] > window[0] * 1.02:
                gamma = min(gamma * 1.15, gamma_high)

            # Steady progress -> trust it, ease gamma down.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.95

    # --- late-stage refinement clamp (monotone, non-increasing) ----------
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)

    # --- final NaN/range guard ------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))