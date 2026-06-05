import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule for differentiable global placement."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: geometric in time, but dominated by overflow ---
    # Cosine-eased progress gives a gentle hold at high gamma early,
    # then a fast drop near the end for accurate HPWL.
    eased = 0.5 - 0.5 * math.cos(math.pi * progress)
    time_base = gamma_high * (gamma_low / gamma_high) ** eased

    # Overflow is the strongest signal of how "spread" the layout still is.
    # While bins are saturated we want smooth gradients (high gamma);
    # as overflow collapses we sharpen toward true HPWL.
    overflow_base = gamma_low + (gamma_high - gamma_low) * (ov ** 1.2)

    # Blend: early on trust overflow, late on trust the time decay so the
    # schedule still converges even if overflow stalls.
    w = progress
    gamma = (1.0 - w) * overflow_base + w * min(time_base, overflow_base + 1.0)

    # --- adaptive feedback from HPWL trajectory ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h > 0]
        if len(recent) >= 5:
            prev = recent[-6] if len(recent) >= 6 else recent[0]
            best_recent = min(recent[-5:])
            rel_gain = (prev - best_recent) / prev if prev > 0 else 0.0

            # Plateau: nudge gamma down to refine wirelength.
            if rel_gain < 1e-3:
                gamma *= 0.85
            # Divergence (HPWL climbing): smooth gradients back up.
            if recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # --- end-game clamp: guarantee accurate HPWL at convergence ---
    if progress > 0.9:
        gamma = min(gamma, 1.0)
    elif progress > 0.75:
        gamma = min(gamma, 2.0)

    return min(50.0, max(0.01, gamma))