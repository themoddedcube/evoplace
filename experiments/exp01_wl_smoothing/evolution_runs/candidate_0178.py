import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine annealing of the WA-WL smoothing parameter."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: log-cosine glide high -> low ---------------------
    # cos_prog moves 0 -> 1 slowly at the ends, fast in the middle, giving
    # a long high-gamma plateau early (cells spread/cluster) and a gentle
    # low-gamma tail for fine HPWL refinement.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ----------------------------------------------
    # While density overflow is high the placement is still legalizing, so
    # keep gradients smooth (raise gamma). As overflow drains, let gamma
    # fall toward the accurate-HPWL regime. Blend a multiplicative and an
    # additive term so neither dominates at the extremes.
    ov_mult = 0.60 + 1.40 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.50)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_add

    # --- HPWL feedback control ------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Stagnation: progress has stalled -> sharpen (lower gamma) to
            # recover real-HPWL signal and escape the plateau.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> gradients too noisy, smooth more.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent -> nudge sharper to bank accuracy.
            elif last < first * 0.98:
                gamma *= 0.93

    # --- late-stage accuracy ceilings -----------------------------------
    # Force the smoothing low near the end so the reported HPWL reflects the
    # true geometry, but stay a little softer if density is still violated.
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.8)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))