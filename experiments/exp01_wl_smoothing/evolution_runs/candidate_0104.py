import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled gamma anneal with HPWL feedback and end-stage clamps."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_high / gamma_low)  # ~2.77

    # --- Primary driver: overflow-coupled gamma (DREAMPlace-style) ---
    # Log-linear map: gamma_low at ov=0, gamma_high at ov=1. A mildly
    # sub-linear exponent keeps gamma high while cells still overlap and
    # lets it fall off quickly once the placement begins to spread.
    ov_shaped = ov ** 0.85
    gamma_ov = gamma_low * math.exp(log_ratio * ov_shaped)

    # --- Secondary driver: cosine progress anneal (smooth fallback) ---
    # Sane trajectory before overflow stabilizes; guarantees monotone-ish
    # cooling toward the end of the run.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_low * math.exp(log_ratio * (1.0 - cos_prog))

    # Trust overflow more as the run proceeds; lean on the schedule early
    # when overflow is still ~1 and uninformative.
    w_ov = 0.35 + 0.55 * progress
    gamma = w_ov * gamma_ov + (1.0 - w_ov) * gamma_prog

    # --- HPWL feedback: react to plateaus / divergence ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # plateau: improvement stalled -> sharpen for accuracy
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # diverging: HPWL trending up -> smooth gradients to recover
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # steadily improving: nudge sharper to lock in gains
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # --- End-stage clamps: force accurate HPWL once nearly converged ---
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.65
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))