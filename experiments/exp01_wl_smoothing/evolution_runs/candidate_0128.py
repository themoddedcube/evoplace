import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Geometric decay along a cosine-warped progress axis: stays high while
    # cells are still spreading, then bends down smoothly for fine-tuning.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Couple to physical density. While overflow is high the placement is still
    # legalizing, so keep gradients smooth; as bins drain, trust the sharper
    # (more accurate) approximation. Blend a multiplicative and an additive term
    # so neither dominates at the extremes.
    ov_mult = 0.60 + 1.50 * (ov ** 1.20)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.60 * base * ov_mult + 0.40 * ov_add

    # React to the optimization trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: relative improvement stalled -> sharpen to chase HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (rising HPWL) -> smooth gradients to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.40
            # Healthy descent -> nudge sharper to lock in the gain.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Endgame: force accuracy once the layout is essentially settled, but only
    # if density is acceptable; otherwise keep a little smoothing headroom.
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))