import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for differentiable global placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth cosine-eased exponential decay in log-space: high early, low late.
    # Cosine easing keeps gamma high a bit longer while cells are clustering,
    # then drops it for accurate HPWL during fine-tuning.
    ease = 0.5 * (1.0 - math.cos(math.pi * progress))
    log_g = math.log(gamma_high) + (math.log(gamma_low) - math.log(gamma_high)) * ease
    base = math.exp(log_g)

    # Overflow coupling: while density overflow is high, cells still need smooth
    # gradients, so hold gamma up. As overflow collapses, let the anneal dominate.
    # Bounded multiplier in [0.7, 2.5] to avoid runaway gamma (numerical blow-up).
    overflow_factor = 0.7 + 1.8 * (ov ** 1.3)
    gamma = base * overflow_factor

    # HPWL-history feedback (defensive: ignore non-finite / non-positive values).
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and math.isfinite(h) and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: sharpen gamma to chase a more accurate HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Divergence guard: HPWL climbing -> smooth gradients back out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # Late-stage cap: keep approximation accurate without going fully unstable.
    if progress > 0.9:
        gamma = min(gamma, 0.9)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))