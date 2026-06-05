import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-cosine gamma anneal with gentle stagnation control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary anneal: smooth log-space cosine from high -> low over progress.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the true physical signal: while cells are still spread
    # (high overflow) we want smoother gradients, so blend toward an
    # overflow-indexed target. As overflow collapses, gamma follows it down
    # for accurate HPWL. Use a convex combination instead of multiply+add
    # so the result can never blow up.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # Weight overflow more heavily early, schedule more heavily late.
    w_ov = 0.5 * (1.0 - progress) + 0.25
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # Gentle history feedback: detect stagnation / divergence on HPWL.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Plateau: barely improving -> sharpen toward accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

            # Diverging (HPWL rising) -> re-smooth gradients a bit.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.20
            # Steadily improving -> nudge sharper to lock in gains.
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # Late-stage ceilings: force accuracy once layout is essentially fixed.
    if progress > 0.85:
        gamma = min(gamma, 1.3 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))