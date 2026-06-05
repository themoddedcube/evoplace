import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with progress backstop and plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow. While bins are congested (early), keep gamma high
    # for smooth gradients; as overflow collapses, anneal toward accurate HPWL.
    # Geometric interpolation in log-space gives a graceful high->low sweep.
    ov_curve = ov ** 0.85
    gamma_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_curve)

    # Backstop: progress-based geometric decay so gamma still falls even if
    # overflow stalls high. Cosine easing keeps it gentle at both ends.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Take the cooler of the two so neither a stuck overflow nor a slow clock
    # can hold gamma artificially high late in the run.
    gamma = min(gamma_ov, gamma_prog)

    # Blend a little of the overflow signal back in early, where keeping gradients
    # smooth matters most for legalizing the rough layout.
    gamma = 0.7 * gamma + 0.3 * (gamma_low + (gamma_high - gamma_low) * ov_curve)

    # Plateau / divergence response from HPWL trace.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # HPWL climbing -> gradients too noisy, smooth them out.
            if last > first * 1.02:
                gamma *= 1.30
            # Healthy descent -> push toward sharper, more accurate gamma.
            elif last < first * 0.97:
                gamma *= 0.92
            # Flat improvement -> nudge down to escape the plateau via accuracy.
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # Late-stage accuracy ceilings: low overflow means we can afford sharp HPWL.
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))