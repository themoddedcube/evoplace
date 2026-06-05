import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored gamma schedule with cosine backbone and plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Backbone: geometric (log-linear) decay along a cosine-warped clock ---
    # Cosine warp holds gamma high a bit longer early, then drops faster mid-run,
    # which matches the "cluster first, refine late" intuition.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow anchoring ---
    # While density is still spread out (high overflow) we MUST keep gradients smooth,
    # so gamma should stay high almost regardless of the iteration clock.
    # As overflow collapses toward 0 the legalization is essentially done and we
    # want accurate (low-gamma) wirelength gradients.
    ov_floor = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)   # density-driven target
    # Blend the clock-driven backbone with the density-driven anchor. Early on the
    # clock dominates; the anchor prevents collapsing gamma while cells are unplaced.
    w_anchor = 0.45 + 0.20 * progress      # trust density more as the run matures
    gamma = (1.0 - w_anchor) * base + w_anchor * max(base * 0.6, ov_floor)

    # Mild multiplicative coupling so very high overflow nudges gamma up further.
    gamma *= 0.85 + 0.30 * (ov ** 1.2)

    # --- HPWL trajectory feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Plateau: best is barely improving -> sharpen (lower gamma) to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Divergence: wirelength climbing -> gradients too noisy, smooth them.
            if last > first * 1.02:
                gamma *= 1.40
            # Healthy descent: gently sharpen to lock in accuracy.
            elif last < first * 0.97:
                gamma *= 0.92

    # --- Late-stage accuracy ceilings (only once density is under control) ---
    if progress > 0.85:
        ceil = 1.4 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.4 if ov > 0.10 else 1.4)

    # Never let gamma crash to near-zero noise while cells are still overlapping.
    if ov > 0.20:
        gamma = max(gamma, 1.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))