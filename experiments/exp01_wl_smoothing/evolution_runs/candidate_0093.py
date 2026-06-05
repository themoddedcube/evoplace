import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven geometric gamma anneal with accuracy-biased tail."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.4   # lower floor -> sharper, more accurate WL late

    # --- overflow-driven geometric anneal --------------------------------
    # Overflow is the physical state signal: high overflow => cells still
    # overlapped, smooth (high) gamma; low overflow => near-legal, accurate
    # (low) gamma. A progress term keeps the tail annealing even when
    # overflow plateaus around its floor (~0.08 in these runs).
    ov_w = ov ** 0.65
    prog_w = (1.0 - progress) ** 1.3
    w = 0.70 * ov_w + 0.30 * prog_w
    w = min(1.0, max(0.0, w))
    gamma = gamma_low * (gamma_high / gamma_low) ** w

    # --- adaptive feedback from HPWL trajectory --------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            if window[-1] > window[0] * 1.015:       # diverging -> smooth
                gamma *= 1.25
            elif window[-1] < window[0] * 0.99:      # improving -> trust it
                gamma *= 0.93
            else:                                    # plateau -> sharpen
                gamma *= 0.85

    # --- late-stage accuracy push ---------------------------------------
    # Once near-legal, force gamma toward the accurate regime for HPWL
    # fine-tuning; keep a higher ceiling if density is still bad.
    if progress > 0.85:
        gamma = min(gamma, 0.9 if ov > 0.10 else 0.45)
    elif progress > 0.70:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)

    # --- final NaN/range guard ------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))