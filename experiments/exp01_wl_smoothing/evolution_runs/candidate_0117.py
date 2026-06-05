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
    # cells are still clustering, then anneals smoothly toward gamma_low.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the physical signal for "how spread out are we". While bins
    # are still congested keep gamma high; as overflow drains, let it fall.
    ov_mult = 0.6 + 1.5 * (ov ** 1.25)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    # Couple the two signals: only allow deep annealing once BOTH the schedule
    # has progressed and density has actually relaxed. Prevents premature
    # sharpening that traps cells in a poor configuration.
    relax = max(progress, 1.0 - ov)
    floor = gamma_low + (gamma_high - gamma_low) * (1.0 - relax) ** 1.5
    gamma = max(gamma, 0.5 * floor)

    # HPWL-trend feedback: react to stagnation and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: sharpen the approximation to chase a tighter HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: HPWL climbing -> gradients too noisy, smooth back out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.35
            # Healthy descent: nudge sharper to refine.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Late-stage caps force an accurate HPWL approximation for final placement,
    # but stay looser while density has not yet settled.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))