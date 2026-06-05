import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven, progress-annealed gamma schedule for WA-WL placement."""

    gamma_high = 8.0
    gamma_low = 0.5

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- primary driver: overflow (DREAMPlace-style) ---------------------
    # Cells still spread (ov ~ 1) -> smooth high gamma; legalized (ov ~ 0)
    # -> sharp low gamma. Log-linear keeps it strictly positive and smooth.
    ov_gamma = gamma_low * (gamma_high / gamma_low) ** ov

    # --- secondary driver: progress via cosine annealing in log space ----
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- blend (geometric mean: bounded, no additive blow-up) ------------
    # Lean on overflow while it is informative, progress provides the floor.
    w = 0.6
    gamma = (ov_gamma ** w) * (prog_gamma ** (1.0 - w))

    # --- plateau / divergence handling from HPWL trace -------------------
    if hpwl_history and len(hpwl_history) >= 4:
        recent = [h for h in hpwl_history[-6:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 4:
            first = recent[0]
            last = recent[-1]
            if first > 0:
                rel = (last - first) / first
                if rel > 0.01:            # HPWL climbing -> smoother grads
                    gamma *= 1.25
                elif abs(rel) < 1e-3:     # stalled -> sharpen approximation
                    gamma *= 0.85

    # --- late-stage accuracy: drive gamma down to refine true HPWL -------
    if progress > 0.85:
        cap = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, cap)
    elif progress > 0.70:
        cap = 2.5 if ov > 0.10 else 1.2
        gamma = min(gamma, cap)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))