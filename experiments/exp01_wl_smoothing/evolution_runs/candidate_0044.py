import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-decayed gamma schedule for WA-WL placement."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.4

    # Cosine-annealed log-space base: smooth start, gentle landing.
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_hi, log_lo = math.log(gamma_high), math.log(gamma_low)
    base = math.exp(log_lo + (log_hi - log_lo) * cos)

    # Overflow is the true physical signal: cells still spread out => keep
    # gamma high; cells legalized => trust the sharp (low-gamma) HPWL.
    overflow_factor = 0.5 + 2.0 * (ov ** 1.2)

    # Blend schedule-progress with the live overflow signal so the schedule
    # tracks actual placement state rather than iteration count alone.
    gamma = (0.6 * base + 0.4 * base * 1.0) * overflow_factor
    gamma = base * (0.4 + 0.6 * overflow_factor)

    # Adapt to HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        rel_improve = (prev - best_recent) / prev if prev > 0 else 0.0

        # Stagnation: sharpen the approximation to escape the plateau.
        if rel_improve < 1e-3:
            gamma *= 0.65

        # Divergence/oscillation: smooth gradients to restabilize.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.6
        elif recent[-1] > min(recent) * 1.005:
            gamma *= 1.15

    # Late-stage: force accuracy once placement is essentially legal.
    if progress > 0.85:
        cap = 1.0 if ov < 0.1 else 2.0
        gamma = min(gamma, cap)

    return min(50.0, max(0.01, gamma))