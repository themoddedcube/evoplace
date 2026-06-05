import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma with cosine-annealed progress fallback."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Primary driver: density overflow (DREAMPlace-style). Smoothing fades as
    # cells spread out. Smoothstep keeps the response gentle and stable.
    ov_s = ov * ov * (3.0 - 2.0 * ov)
    log_ov = log_lo + (log_hi - log_lo) * ov_s

    # Secondary driver: cosine annealing on progress. Guarantees gamma decays
    # toward gamma_low even if overflow plateaus mid-run.
    cos_t = 0.5 * (1.0 - math.cos(math.pi * progress))
    log_pr = log_hi + (log_lo - log_hi) * cos_t

    # Trust overflow early (cells clustered), trust the schedule late.
    w = progress
    gamma = math.exp((1.0 - w) * log_ov + w * log_pr)

    # Respond to the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        # Stalled improvement -> sharpen approximation to refine wirelength.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85
        # Worsening / oscillating -> smooth the gradients.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.3

    # Final refinement: enforce accurate HPWL approximation near the end.
    if progress > 0.8:
        gamma = min(gamma, gamma_low * 2.0)
    if progress > 0.92:
        gamma = min(gamma, gamma_low)

    return min(50.0, max(0.01, gamma))