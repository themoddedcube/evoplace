import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with a progress anchor and HPWL feedback."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary driver: overflow-adaptive exponential (DREAMPlace-style) ---
    # Congested layout (high overflow) -> large gamma for smooth gradients;
    # as cells spread (overflow -> 0) gamma shrinks toward gamma_low for an
    # accurate HPWL approximation. Geometric interpolation in log-space.
    gamma_ov = gamma_low * (gamma_high / gamma_low) ** ov

    # --- secondary driver: cosine progress decay as an annealing anchor ---
    # Guarantees the schedule keeps sharpening even if the overflow signal is
    # noisy or saturates high early in the run.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Trust overflow most, but keep the progress anchor so we never stay smooth
    # forever on a high-overflow plateau.
    w = 0.7
    gamma = w * gamma_ov + (1.0 - w) * gamma_prog

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-7:] if (h is not None and h == h and h > 0)]
        if len(recent) >= 6:
            window = recent[-5:]
            prev = recent[-6]
            best_recent = min(window)
            # stagnation -> sharpen approximation
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.9
            # divergence (HPWL climbing) -> smooth more to stabilize
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage refinement cap ---
    if progress > 0.85:
        ceil = 2.0 if ov > 0.10 else 1.0
        gamma = min(gamma, ceil)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))