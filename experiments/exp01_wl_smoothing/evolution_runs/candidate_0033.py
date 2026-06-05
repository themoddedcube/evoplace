import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth cosine-shaped interpolation in log-space: stays high while cells
    # are still spreading, then decays quickly once placement settles.
    cos_w = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_g = math.log(gamma_low) + (math.log(gamma_high) - math.log(gamma_low)) * cos_w
    base = math.exp(log_g)

    # Overflow is the primary physical signal: when bins are still congested we
    # need smoother gradients; once density relaxes we can sharpen toward HPWL.
    overflow_factor = 0.55 + 2.2 * (ov ** 1.3)

    # Blend the time-based anneal with the overflow signal so that a stalled
    # spreading phase keeps gamma high even if iterations advance.
    gamma = base * (0.35 + 0.65 * overflow_factor)

    # Adapt to the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        # Plateau: relative improvement stalled -> sharpen toward true HPWL.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.65
        # Divergence: HPWL climbing -> smooth gradients to recover stability.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.6
        # Strong, healthy descent -> let it push a bit harder on accuracy.
        elif recent[0] > 0 and recent[-1] < recent[0] * 0.98:
            gamma *= 0.9

    # Late stage must converge on accurate wirelength regardless of history.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))