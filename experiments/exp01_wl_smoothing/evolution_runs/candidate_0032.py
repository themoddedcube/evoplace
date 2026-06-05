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

    # Cosine-annealed exponential decay: smooth high->low transition that
    # holds high gamma a touch longer early, then accelerates the descent.
    cos_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    blend = 0.5 * progress + 0.5 * cos_progress
    base = gamma_high * (gamma_low / gamma_high) ** blend

    # Overflow coupling: physical spreading state drives smoothness more than
    # raw iteration count. When bins are saturated keep gradients smooth; once
    # the layout has spread, let gamma fall toward the accurate regime.
    overflow_factor = 0.5 + 1.9 * (ov ** 1.3)
    gamma = base * overflow_factor

    # HPWL feedback: react to convergence dynamics.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        finite = [h for h in recent if h == h and abs(h) != float("inf")]

        if finite and prev > 0:
            rel_gain = (prev - best_recent) / prev
            # Plateau: sharpen the approximation to chase HPWL detail.
            if rel_gain < 1e-3:
                gamma *= 0.65
            # Strong steady improvement: hold current smoothness.
            elif rel_gain > 2e-2:
                gamma *= 1.05

        # Oscillation/divergence guard: re-smooth to stabilize gradients.
        if recent[-1] > recent[0] * 1.02:
            gamma *= 1.6

    # Late-stage cap: force accurate regime for final fine-tuning.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.2)

    if gamma != gamma:  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))