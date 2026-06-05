import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-interpolated gamma with progress fallback and
    trend-aware correction. High gamma while spread out (high overflow /
    early), low gamma once converged for accurate HPWL."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Primary driver: overflow reflects the true spreading state.
    # Map ov in [0,1] -> weight in [0,1] (concave so gamma stays usefully
    # high until cells are genuinely well spread).
    w_ov = ov ** 0.85

    # Secondary driver: cosine annealing on progress as a fallback so gamma
    # keeps decaying even if overflow plateaus.
    w_prog = 1.0 - (0.5 - 0.5 * math.cos(math.pi * progress))

    # Blend in log-space, weighting overflow more heavily.
    w = 0.65 * w_ov + 0.35 * w_prog
    gamma = math.exp(log_lo + (log_hi - log_lo) * w)

    # Trend-aware correction from recent HPWL history.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            if window[-1] > window[0] * 1.02:        # diverging -> smooth more
                gamma *= 1.25
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85                         # stalled -> sharpen
            elif window[-1] < window[0] * 0.995:      # improving -> sharpen mildly
                gamma *= 0.93

    # Late-stage caps to enforce accurate HPWL once mostly converged.
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))