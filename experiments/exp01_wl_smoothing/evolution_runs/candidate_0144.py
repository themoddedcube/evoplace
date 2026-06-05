import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress prior and HPWL feedback."""

    gamma_high = 8.0
    gamma_low = 0.5

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # Core driver: overflow measures spatial state directly. High overflow ->
    # cells still spreading/clustering -> high gamma (smooth gradients). Low
    # overflow -> spatially converged -> low gamma (accurate HPWL). Geometric
    # interpolation keeps the transition multiplicative/smooth.
    ov_gamma = gamma_low * (gamma_high / gamma_low) ** ov

    # Progress prior: cosine annealing from high to low. Acts as a fallback
    # when overflow is uninformative (very early, or stuck) and damps the noise
    # in the raw overflow signal.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: early iterations trust the schedule prior (overflow ~1 and noisy);
    # later iterations trust the measured overflow, which reflects real state.
    w = progress
    gamma = (1.0 - w) * prog_gamma + w * ov_gamma
    # Never let the prior pull gamma far below what overflow demands: if many
    # bins are still over-dense we must keep gradients smooth regardless of time.
    gamma = max(gamma, 0.6 * ov_gamma)

    # HPWL feedback: react to the actual optimization trajectory.
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 4:
            first = recent[0]
            last = recent[-1]
            best_recent = min(recent)
            # Diverging: HPWL climbing -> gradients too noisy -> smooth more.
            if last > first * 1.01:
                gamma *= 1.25
            # Plateaued: little progress and not diverging -> sharpen to refine.
            elif first > 0 and (first - best_recent) / first < 1e-3:
                gamma *= 0.82

    # Late-stage commitment: once spatially converged, force accurate HPWL.
    if progress > 0.85:
        ceil = 0.8 if ov < 0.10 else 1.5
        gamma = min(gamma, ceil)
    elif progress > 0.70 and ov < 0.10:
        gamma = min(gamma, 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))