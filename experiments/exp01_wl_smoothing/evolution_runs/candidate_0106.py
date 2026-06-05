import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space annealing from high to low gamma along progress.
    # Cosine warm-in keeps gamma high while cells are still spreading,
    # then accelerates the descent in the back half for accurate HPWL.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the true physical signal of how clustered the design is.
    # When density overflow is high, gradients must stay smooth (high gamma);
    # as the layout legalizes (ov -> 0) we trust low gamma for sharp HPWL.
    # Blend a multiplicative term (scales the annealed base) with an additive
    # floor/ceiling driven directly by overflow so neither dominates.
    ov_mult = 0.70 + 0.90 * (ov ** 1.15)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.6 * base * ov_mult + 0.4 * ov_add

    # Adaptive feedback from recent HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0 and h != float('inf')]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            denom = window[0] if window[0] > 0 else 1.0

            # Plateau: progress stalled -> sharpen toward accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Divergence: HPWL climbing -> back off, re-smooth gradients.
            if window[-1] > denom * 1.02:
                gamma *= 1.30
            # Healthy descent: nudge sharper to lock in gains.
            elif window[-1] < denom * 0.98:
                gamma *= 0.93

    # Late-stage caps: force convergence to accurate regime, but keep a
    # higher ceiling while overflow remains (legalization not yet done).
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma or gamma in (float('inf'), float('-inf')):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))