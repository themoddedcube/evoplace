import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress floor and stagnation control."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: overflow. DREAMPlace-style coupling — gamma stays high
    # while bins are congested (cells still spreading), collapses as overflow
    # drops and the placement settles. Smooth exponential map of overflow.
    ov_term = gamma_low * (gamma_high / gamma_low) ** ov

    # Secondary driver: time progress provides a monotone decay floor so gamma
    # keeps annealing even if overflow plateaus.
    prog_term = gamma_high * (gamma_low / gamma_high) ** progress

    # Blend: early rely on progress, late rely on overflow accuracy signal.
    w = progress
    gamma = (1.0 - w) * prog_term + w * ov_term

    # Adaptive response to HPWL trajectory.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: sharpen approximation to chase finer wirelength.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Divergence / oscillation: smooth gradients to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.4

    # Late-stage accuracy cap, but keep a small floor so gradients persist.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.75:
        gamma = min(gamma, 1.5)

    if not math.isfinite(gamma):
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))