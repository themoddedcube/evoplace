import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Anneal gamma from high (smooth gradients) to low (accurate HPWL),
    driven primarily by overflow with a progress-based backstop and
    HPWL-trajectory adaptation."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Overflow is the physical signal: ~1.0 while cells overlap, -> 0 as the
    # layout legalizes. Interpolate gamma in log-space on a shaped overflow so
    # smoothness tracks the real spreading state, not merely elapsed time.
    ov_s = ov ** 0.7  # hold gamma high while overlap is large, decay fast late
    gamma = math.exp(log_lo + (log_hi - log_lo) * ov_s)

    # Monotone time backstop: ensures gamma still descends if overflow stalls
    # high. Blend in log-space, weighting the overflow term more heavily.
    time_gamma = math.exp(log_hi + (log_lo - log_hi) * progress)
    w = 0.65
    gamma = math.exp(w * math.log(gamma) + (1.0 - w) * math.log(time_gamma))

    # Adapt to the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        # Plateau: sharpen the approximation to chase true wirelength.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85
        # Divergence: smooth gradients back out to recover stability.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.3

    # Late-stage fine-tuning regime: accurate but not maximally noisy.
    if progress > 0.9:
        gamma = min(gamma, 0.8)

    return min(50.0, max(0.01, gamma))