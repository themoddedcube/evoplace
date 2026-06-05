import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma with progress annealing and HPWL-plateau adaptation.

    Primary signal is overflow (the proven DREAMPlace driver): gamma interpolates
    log-linearly between a low and high bound as overflow goes 0 -> 1. Progress
    pulls the effective bounds down over time so late iterations fine-tune HPWL
    even if overflow lingers. HPWL feedback nudges around plateaus / divergence.
    """

    # ---- sanitize inputs ----
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # ---- progress-annealed bounds ----
    # Late in placement, even high overflow should not demand maximal smoothing,
    # and the floor drifts toward the accurate (low-gamma) regime for fine HPWL.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)  # smooth 0 -> 1
    eff_log_high = log_high - (log_high - log_low) * 0.55 * cos_prog
    eff_log_low = log_low - 0.5 * cos_prog  # allow dipping a touch below 0.5 late

    # ---- overflow drives position between the bounds (log space) ----
    # Emphasize the high end while cells are still spread out (overflow high),
    # collapse quickly toward accurate gamma as overflow resolves.
    ov_shape = ov ** 1.3
    log_gamma = eff_log_low + (eff_log_high - eff_log_low) * ov_shape
    gamma = math.exp(log_gamma)

    # ---- HPWL feedback ----
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: stalled improvement -> sharpen toward accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging: HPWL trending up -> add smoothing to regain stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Healthy descent -> stay slightly accurate.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.96

    # ---- late-stage ceilings to lock in accurate HPWL ----
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # ---- final guard ----
    if gamma != gamma:  # NaN
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))