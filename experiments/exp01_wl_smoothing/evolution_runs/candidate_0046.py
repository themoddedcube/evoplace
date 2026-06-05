import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule: high gamma early (smooth
    gradients, cells cluster) decaying to low gamma late (accurate
    HPWL, fine-tuning). Overflow drives the primary anneal so that
    gamma stays high while bins remain congested and only drops once
    density actually relaxes; iteration progress is a soft fallback."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Geometric anneal in log-space. Primary driver is overflow (the
    # true measure of how clustered cells are); progress is a gentle
    # backstop so gamma still descends if overflow plateaus high.
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # Blend an overflow-driven anneal with a progress-driven one.
    # ov_anneal: 1.0 when fully congested -> gamma_high, 0.0 -> gamma_low.
    ov_anneal = ov ** 0.85
    prog_anneal = 1.0 - progress
    # Weight overflow more heavily but keep progress influential late.
    w = 0.7
    anneal = w * ov_anneal + (1.0 - w) * prog_anneal
    anneal = min(1.0, max(0.0, anneal))

    gamma = math.exp(log_lo + (log_hi - log_lo) * anneal)

    # Plateau / divergence response from HPWL trend.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            # Stalled improvement: sharpen the approximation to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.8
            # Diverging wirelength: smooth gradients to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # Final fine-tuning phase: force an accurate, low-gamma regime.
    if progress > 0.85:
        gamma = min(gamma, 1.0)

    if not (gamma == gamma) or gamma == float("inf"):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))