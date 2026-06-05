import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-coupled annealed gamma schedule for differentiable global placement."""

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing --------------------------------------------------
    # Smooth (cosine-eased) geometric interpolation in log-space from
    # gamma_high -> gamma_low. Geometric decay keeps relative steps even,
    # which suits the multiplicative WA-WL smoothing parameter.
    ease = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** ease

    # --- overflow coupling ----------------------------------------------
    # Physical state (overflow) is more trustworthy than the iteration
    # clock: while cells are still badly overlapping we must keep gamma
    # high regardless of progress; once spread out we let it drop. Blend
    # an overflow-driven target with the schedule so neither dominates.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # weight toward overflow target early, toward annealed base late
    w_ov = 0.45 * (1.0 - progress) + 0.15
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # gentle multiplicative nudge so very-full / very-empty states are
    # pushed further in the expected direction without runaway
    gamma *= 0.75 + 0.5 * ov

    # --- adaptive feedback from HPWL trajectory --------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # diverging / oscillating up -> smooth more (raise gamma)
            if last > first * 1.02:
                gamma *= 1.30
            # steady improvement -> sharpen (lower gamma) to refine HPWL
            elif last < first * 0.985:
                gamma *= 0.92
            # plateau (no meaningful gain) -> nudge sharper to escape
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # --- late-stage HPWL-accuracy ceilings ------------------------------
    # Near the end we want accurate (low-gamma) wirelength, but only once
    # density is acceptable; otherwise keep some smoothing to finish legal.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))