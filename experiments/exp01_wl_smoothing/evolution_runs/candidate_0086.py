import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress-based annealing and
    light HPWL-feedback, returning a value in [0.01, 50.0]."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary signal: overflow ---
    # DREAMPlace-style coupling: gamma tracks how spread-out the cells are.
    # When bins are full (ov->1) keep gamma high for smooth gradients; as the
    # layout settles (ov->0) drop gamma for an accurate HPWL approximation.
    # Geometric interpolation in log-space gives a smooth high->low sweep.
    ov_curve = ov ** 0.75
    overflow_gamma = gamma_low * (gamma_high / gamma_low) ** ov_curve

    # --- secondary signal: training progress (cosine annealing) ---
    # Guarantees monotone-ish cooling even if overflow plateaus, so late
    # iterations never get stuck at a high, inaccurate gamma.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    progress_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: lean on overflow early, on progress late.
    w = progress
    gamma = (1.0 - w) * overflow_gamma + w * progress_gamma

    # --- HPWL feedback (gentle, bounded) ---
    if hpwl_history:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 4:
            window = recent[-4:]
            best_recent = min(window)
            head = window[0]
            # Stalled improvement: sharpen the approximation to refine wirelength.
            if head > 0 and (head - best_recent) / head < 1e-3:
                gamma *= 0.85
            # Diverging HPWL: smooth gradients back out to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage ceiling: force accurate HPWL near the end ---
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)

    if not (gamma == gamma):  # NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))