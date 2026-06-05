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
    gamma_low = 0.5

    # Cosine-annealed schedule in log-space: smooth high->low transition that
    # spends more iterations near the low (accurate) end than pure geometric decay.
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    base = math.exp(log_lo + (log_hi - log_lo) * cos_factor)

    # Overflow coupling: when cells are still spread/over-dense keep gamma high
    # for smoother gradients; as density resolves, let gamma fall toward base.
    overflow_factor = 0.55 + 1.85 * (ov ** 1.3)
    gamma = base * overflow_factor

    # HPWL feedback: react to convergence dynamics.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # Plateau: sharpen approximation to chase lower HPWL.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.65

        # Divergence/oscillation: smooth gradients back out.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.6

        # Strong steady improvement: gently push sharper to keep gaining.
        if prev > 0 and (prev - recent[-1]) / prev > 1e-2:
            gamma *= 0.9

    # Final refinement window: enforce accurate (low) gamma for fine placement.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    return min(50.0, max(0.01, gamma))