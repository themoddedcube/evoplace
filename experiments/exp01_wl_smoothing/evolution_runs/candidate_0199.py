import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with smooth exponential annealing.

    Strategy: keep gamma high while cells are still spread out (high
    overflow / early progress) for smooth gradients, then anneal toward a
    small gamma for an accurate HPWL approximation during fine-tuning.
    Overflow is the primary driver; iteration progress is a fallback so the
    schedule still cools down even if the overflow signal stalls.
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
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # --- overflow-driven annealing (primary signal) ------------------------
    # When overflow is high (cells spread out) we want gamma near gamma_high;
    # as the design legalizes (overflow -> 0) we anneal toward gamma_low.
    # The exponent shapes the descent so gamma stays high for longer while
    # there is meaningful overflow, then drops quickly near the end.
    ov_shape = ov ** 0.65
    log_ov = log_low + (log_high - log_low) * ov_shape

    # --- progress-driven cosine annealing (fallback signal) ----------------
    # Guards against a stalled/uninformative overflow value by guaranteeing a
    # monotone cooldown over the run.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_prog = log_high + (log_low - log_high) * cos_prog

    # Blend: trust overflow more, but let progress pull gamma down late.
    w_prog = 0.30 + 0.40 * progress
    log_gamma = (1.0 - w_prog) * log_ov + w_prog * log_prog
    gamma = math.exp(log_gamma)

    # --- HPWL-history feedback (gentle, well-bounded) ----------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first = window[0]
            last = window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau in HPWL improvement: push gamma down to sharpen the
            # approximation and escape the flat region.
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence (HPWL climbing): smooth the gradients back out.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent: nudge toward accuracy.
            elif last < first * 0.98:
                gamma *= 0.95

    # --- end-of-run accuracy ceilings -------------------------------------
    # Late in the run an accurate HPWL approximation matters most, so cap
    # gamma unless overflow is still high (legalization not yet achieved).
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guards ------------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))