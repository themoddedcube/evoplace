import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress decay and plateau adaptation.

    Primary driver is overflow (DREAMPlace-style): gamma spans orders of
    magnitude as the layout decongests. Progress and HPWL-history terms
    provide secondary fine-tuning and stabilization. Fully NaN/edge guarded.
    """

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

    # --- primary: overflow-adaptive exponential mapping ---
    # log-space interpolation keyed on overflow. When the layout is congested
    # (ov high) gamma is large and gradients are smooth; as bins decongest
    # gamma collapses toward gamma_low for an accurate HPWL approximation.
    # ov is reshaped (ov**0.85) so gamma drops only once overflow is genuinely low.
    ov_key = ov ** 0.85
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_key)

    # --- secondary: cosine progress floor ---
    # Independently anneal with iteration so gamma still descends even if the
    # overflow estimate stalls. Combined as a geometric blend (stays positive).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    gamma = (gamma_ov ** 0.6) * (gamma_prog ** 0.4)

    # --- HPWL-history plateau / divergence adaptation ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # diverging: HPWL climbing -> too noisy, smooth gradients back up
            if last > first * 1.02:
                gamma *= 1.30
            # plateaued improvement: sharpen approximation to refine HPWL
            elif prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80
            # healthy descent: gentle sharpening
            elif last < first * 0.98:
                gamma *= 0.94

    # --- late-stage ceilings: force accuracy near the end ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    # --- final guards ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))