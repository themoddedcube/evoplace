import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled gamma schedule for WA-WL global placement.

    High gamma while cells are still overlapping (high overflow / early
    progress) for smooth gradients, annealing toward low gamma for an
    accurate HPWL approximation once the layout has spread out. Overflow
    is treated as the primary driver and time-progress as a soft guide,
    with HPWL-history feedback for plateau/divergence handling.
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

    # --- overflow-driven base (geometric interpolation in log-space) -----
    # When overflow is high we want gamma near gamma_high; as overflow
    # drains toward 0 we want gamma near gamma_low. A mild concave power
    # keeps gamma elevated until overflow genuinely starts to clear.
    ov_eff = ov ** 0.85
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_eff)

    # --- time-driven base (cosine anneal in log-space) -------------------
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_time = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- blend: overflow leads early, time guides the tail ---------------
    # Trust overflow more while it is informative (still high); lean on the
    # time schedule late so we commit to fine-tuning even if overflow lingers.
    w_time = 0.30 + 0.45 * progress
    gamma = (1.0 - w_time) * gamma_ov + w_time * gamma_time

    # --- HPWL-history feedback ------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: progress has stalled -> sharpen (lower gamma) to chase
            # a more accurate objective, but only modestly.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth gradients back up.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Healthy descent -> nudge gamma down to tighten approximation.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # --- late-stage accuracy ceilings -----------------------------------
    # Force an accurate HPWL approximation near the end, but stay higher
    # if density has not yet legalized (overflow still meaningful).
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.80:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 1.8)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))