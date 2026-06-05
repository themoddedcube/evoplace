import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for WA-WL smoothing in differentiable
    global placement. High gamma while cells are spread (high overflow / early),
    decaying smoothly to low gamma for accurate HPWL during fine-tuning."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- primary driver: overflow (the physically meaningful signal) ---
    # DREAMPlace-style coupling: gamma tracks how spread the placement is.
    # Smooth, monotone map ov in [0,1] -> gamma in [low, high] in log space.
    ov_curve = ov ** 0.9                           # slightly front-loaded
    g_ov = math.exp(log_lo + (log_hi - log_lo) * ov_curve)

    # --- secondary driver: time progress via cosine annealing in log space ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    g_prog = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # Blend: rely on overflow more early, on the time floor more late so the
    # schedule still anneals even if overflow plateaus.
    w_ov = 0.65 - 0.30 * progress
    gamma = math.exp(w_ov * math.log(g_ov) + (1.0 - w_ov) * math.log(g_prog))

    # --- HPWL-history feedback: gentle, bounded adjustments ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau in best HPWL -> sharpen (lower gamma) to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging (HPWL rising) -> smooth more to recover stability.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> nudge toward accuracy.
            elif last < first * 0.97:
                gamma *= 0.93

    # --- late-stage accuracy ceilings (allow more smoothing if still spread) ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)

    if gamma != gamma:                             # final NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))