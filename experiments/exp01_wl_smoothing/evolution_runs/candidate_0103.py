import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-log gamma anneal with stagnation control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth log-space anneal: stays high while cells cluster, drops late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow is the true signal of spreading: keep gamma high while bins are
    # congested, regardless of where we are in the iteration budget.
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.5 * base + 0.5 * ov_floor
    gamma *= 0.7 + 0.9 * (ov ** 1.1)

    # HPWL trajectory feedback.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0] if window[0] > 0 else 1.0

            # Plateau: anneal faster to sharpen the HPWL approximation.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging: smooth gradients back out.
            if window[-1] > first * 1.02:
                gamma *= 1.30
            # Improving steadily: let it keep sharpening.
            elif window[-1] < first * 0.98:
                gamma *= 0.92

    # Late-stage ceilings: once spread is mostly resolved, force accuracy.
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))