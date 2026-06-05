import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for differentiable global placement.

    Strategy: gamma is driven primarily by *overflow* (the physical state of the
    layout) rather than raw iteration count, because overflow is the true signal
    of how clustered/legal the placement is. Progress is used only as a gentle
    annealing backbone and a late-stage safety cap. A plateau/divergence detector
    on hpwl_history nudges gamma to escape stalls or damp oscillations.
    """

    gamma_high = 8.0
    gamma_low = 0.5

    # --- sanitize inputs ---------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- overflow-driven base ---------------------------------------------
    # Geometric interpolation in log-space between gamma_high and gamma_low.
    # When overflow is high (cells overlapping) we want a large, smoothing gamma;
    # as bins empty out we sharpen toward gamma_low for accurate HPWL.
    # ov**0.4 makes gamma stay high until overflow is genuinely small, which
    # protects HPWL accuracy from kicking in before the layout is spread out.
    ov_key = ov ** 0.4
    ov_gamma = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_key)

    # --- progress annealing backbone --------------------------------------
    # Cosine-in-log annealing as a fallback when overflow is uninformative
    # (e.g. early before density settles). Blend the two, weighting overflow
    # more as the run advances and the overflow signal becomes trustworthy.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog

    w_ov = 0.35 + 0.45 * progress               # 0.35 -> 0.80
    gamma = w_ov * ov_gamma + (1.0 - w_ov) * prog_gamma

    # --- HPWL trend adaptation --------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau: best HPWL barely improving -> sharpen to refine detail.
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> smooth harder to regain stability.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent: trust the trajectory, sharpen slightly.
            elif last < first * 0.985:
                gamma *= 0.93

    # --- late-stage caps for accurate final HPWL --------------------------
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp -------------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))