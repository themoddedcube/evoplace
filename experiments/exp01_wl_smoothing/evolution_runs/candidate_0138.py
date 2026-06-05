import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement.

    High gamma early (smooth gradients, clustered cells) decaying to low
    gamma late (accurate HPWL, fine placement). Overflow modulates the
    decay so we stay smooth while density is still high, and a light
    plateau/divergence guard nudges gamma when progress stalls.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    if ov == float("inf") or ov == float("-inf"):
        ov = 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base geometric (log-linear) cosine anneal -----------------------
    # cos_prog ramps 0 -> 1 smoothly; geometric interpolation keeps the
    # multiplicative spacing natural for an exponential-like quantity.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ----------------------------------------------
    # While overflow is high, cells are still spreading: bias gamma up so
    # gradients stay smooth. As overflow collapses, let base dominate.
    ov_mult = 0.50 + 1.50 * (ov ** 1.3)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.6)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_add

    # --- plateau / divergence response ----------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0] if window[0] > 0 else 1.0

            # stalled improvement -> sharpen (lower gamma) to refine
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # diverging -> smooth (raise gamma) to recover
            if window[-1] > first * 1.02:
                gamma *= 1.30
            elif window[-1] < first * 0.98:
                gamma *= 0.93

    # --- late-stage ceilings (lock in accurate HPWL) --------------------
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))