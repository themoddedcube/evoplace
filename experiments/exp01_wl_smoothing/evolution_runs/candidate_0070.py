import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma annealing with trend-aware correction.

    High gamma while cells are clustered (high overflow / early), low gamma for
    accurate HPWL once spread out (low overflow / late). Combines a physical
    overflow-driven term with a progress-driven floor so gamma always descends
    even if overflow stalls, then nudges on HPWL trend.
    """
    gamma_high = 8.0
    gamma_low = 0.5

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = min(1.0, max(0.0, iteration / total))

    ov = overflow if (overflow is not None and math.isfinite(overflow)) else 1.0
    ov = min(1.0, max(0.0, ov))

    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Overflow-driven term: smooth cosine ramp in log-space.
    # weight -> 1 when fully overflowed, -> 0 when spread out.
    w_ov = 0.5 * (1.0 - math.cos(math.pi * ov))
    log_ov = log_lo + (log_hi - log_lo) * w_ov

    # Progress-driven term: monotone exponential decay, guarantees descent.
    log_prog = log_hi + (log_lo - log_hi) * progress

    # Trust overflow early, schedule late (geometric blend).
    blend = progress
    gamma = math.exp((1.0 - blend) * log_ov + blend * log_prog)

    # Trend-aware correction from HPWL history.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-5:]
                  if h is not None and math.isfinite(h) and h > 0]
        if len(recent) >= 3:
            if recent[-1] > recent[0] * 1.01:
                # HPWL rising -> placement degrading -> smooth more.
                gamma *= 1.3
            else:
                rel = (recent[0] - min(recent)) / recent[0]
                if rel < 1e-3:
                    # Converged plateau -> sharpen for accuracy.
                    gamma *= 0.85

    # Late-stage cap to lock in accurate HPWL.
    if progress > 0.9:
        gamma = min(gamma, 1.0)

    return min(50.0, max(0.01, gamma))