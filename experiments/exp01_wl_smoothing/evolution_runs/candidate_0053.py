import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive gamma schedule for WA-WL smoothing."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Smooth cosine-annealed exponential base: stays high while cells cluster,
    # decays gently to gamma_low for accurate HPWL near the end.
    cos_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    base = gamma_high * (gamma_low / gamma_high) ** cos_progress

    # Overflow is the physical signal for "how clustered are we": when bins are
    # still congested we want smooth gradients (high gamma); as overflow drops
    # we trust HPWL more and sharpen. Blend with schedule progress so we never
    # collapse gamma while overflow is still high.
    overflow_factor = 0.5 + 1.5 * (ov ** 1.2)
    gamma = base * overflow_factor

    # Adaptive feedback from the HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)

        # Stagnation: relative improvement over the window is tiny -> sharpen
        # gradients to escape the plateau and refine wirelength.
        if prev > 0 and (prev - best_recent) / prev < 1e-3:
            gamma *= 0.75

        # Divergence: HPWL trending up -> smooth gradients to re-stabilize.
        if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
            gamma *= 1.4

    # Late-stage cap: once placement is mostly legal, force accurate HPWL.
    if progress > 0.85:
        cap = 1.0 if ov < 0.15 else 2.0
        gamma = min(gamma, cap)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))