import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven cosine gamma schedule with plateau adaptation."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # --- base annealing in log-space -------------------------------------
    # Cosine schedule: stays high early, decays smoothly, flattens late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)

    # Overflow is the physically meaningful signal: while bins are still
    # congested we must keep gradients smooth (high gamma); once spread out
    # we can sharpen. Blend a progress-driven term with an overflow-driven
    # term so the schedule self-adapts to the actual placement state.
    ov_term = ov ** 0.85               # high while congested, ->0 when spread
    blend = 0.55 * cos_prog + 0.45 * (1.0 - ov_term)
    blend = min(1.0, max(0.0, blend))

    log_gamma = log_high + (log_low - log_high) * blend
    gamma = math.exp(log_gamma)

    # --- plateau / divergence adaptation from HPWL history ---------------
    if hpwl_history:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            ref = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stalled improvement: sharpen to refine HPWL, but only once the
            # density has settled enough that sharper gradients are safe.
            if ref > 0.0 and (ref - best_recent) / ref < 1e-3 and ov < 0.15:
                gamma *= 0.80

            # Diverging HPWL: back off toward smoother gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage refinement ceiling -----------------------------------
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)

    if not (gamma == gamma):  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))