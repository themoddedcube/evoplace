import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-annealed gamma for WA-WL smoothing.

    Overflow is the true convergence state, so it -- not raw progress --
    drives the schedule; progress only adds a gentle late-stage taper.
    Keying on overflow keeps behaviour consistent across the short/medium/
    long cascade stages (which have very different total_iterations) and
    structurally avoids the low-gamma-while-still-spread regime that
    produces noisy gradients and divergence.
    """

    # --- sanitize inputs ------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:            # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- overflow-driven log interpolation (primary signal) ------------
    # smoothstep(ov) keeps gamma high while density is unresolved and
    # lets it collapse only as the layout actually clusters.
    s = ov * ov * (3.0 - 2.0 * ov)
    gamma = gamma_low * (gamma_high / gamma_low) ** s

    # --- progress taper (secondary) ------------------------------------
    # Late in a run, bias toward the accurate (low) regime even if
    # overflow is slightly elevated -- bounded so it can't dominate.
    gamma *= (1.0 - 0.35 * progress)

    # --- adaptive feedback from HPWL trajectory ------------------------
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)
            # HPWL climbing -> smooth harder to re-stabilize.
            if window[-1] > window[0] * 1.01:
                gamma *= 1.30
            # Plateaued *and* already fairly dense -> sharpen for accuracy.
            elif ov < 0.25 and prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # --- safety floor: never go noisy while cells are still spread -----
    # This is the guard the Stage-0 gate (norm_hpwl < 2.0) rejects when
    # absent: too-low gamma at high overflow never clusters and diverges.
    floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = max(gamma, 0.6 * floor)

    # --- late-stage refinement clamp -----------------------------------
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final NaN/range guard -----------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))