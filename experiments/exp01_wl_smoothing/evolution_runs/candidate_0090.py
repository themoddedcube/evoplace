import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL smoothing."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: smooth cosine-warped geometric decay ------------
    # cos warp keeps gamma high a bit longer early, then decays fast late.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ----------------------------------------------
    # While cells are still spread (high overflow) we want smoother
    # gradients, so bias gamma upward; as overflow collapses, let it fall
    # toward the accurate low-gamma regime. Blend multiplicatively and
    # additively for stability.
    ov_mult = 0.55 + 1.6 * (ov ** 1.25)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)
    gamma = 0.55 * base * ov_mult + 0.45 * ov_add

    # --- adaptive feedback from HPWL trajectory --------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: improvement stalled -> sharpen (lower gamma) to
            # chase a more accurate wirelength gradient.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.82

            # Divergence: HPWL climbing -> smooth out (raise gamma) to
            # stabilize the optimization.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.35

            # Strong steady progress -> trust it, ease gamma down gently.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.95

    # --- late-stage refinement clamp ------------------------------------
    # Near the end, force accurate (low) gamma unless density is still bad.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final NaN/range guard ------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))