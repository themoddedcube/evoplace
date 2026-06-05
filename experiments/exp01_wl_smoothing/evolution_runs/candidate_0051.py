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

    # Smooth geometric anneal from high -> low over the run (time-driven floor).
    base = gamma_high * (gamma_low / gamma_high) ** progress

    # Overflow is the true placement-quality signal: keep gamma high while cells
    # are still spread out, ease it down as the layout legalizes. Blend the
    # time-anneal with an overflow-anneal so neither dominates pathologically.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.3)
    gamma = 0.5 * base + 0.5 * (0.5 * base + 0.5 * ov_target)

    # Adapt to optimization dynamics, but bound the multipliers so a single bad
    # window can never blow gamma up (which is what diverges HPWL to inf).
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # Stalled improvement: sharpen the approximation to fine-tune.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.85

        # Rising HPWL (instability): smooth gradients a little, but gently.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.2

    # Late-stage cap: prioritize accurate HPWL once mostly converged.
    if progress > 0.85:
        gamma = min(gamma, 1.0)
    if progress > 0.95:
        gamma = min(gamma, gamma_low)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))