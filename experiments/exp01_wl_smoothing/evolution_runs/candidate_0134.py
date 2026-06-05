import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma for differentiable global placement.

    High gamma early for smooth, stable gradients while cells cluster;
    smooth exponential/cosine decay toward low gamma for accurate HPWL
    as the layout settles. Overflow gates the floor so we never sharpen
    faster than the density actually resolves, and gentle plateau /
    divergence handling nudges (never slams) the smoothness.
    """

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

    # --- smooth annealing core ------------------------------------------
    # Cosine-shaped progress => slow start, slow finish, fast middle.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    # Geometric interpolation in log-space is stable and monotone.
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow gating -------------------------------------------------
    # When density is still high we must NOT sharpen (low gamma + noisy
    # gradients would scatter cells); when density has resolved we are
    # free to drop gamma for an accurate HPWL approximation.
    # ov ~ 0.08 in this regime => gate ~0.25, so we lean accurate.
    gate = ov ** 0.5
    gamma = base * (0.35 + 0.65 * gate) + gamma_low * (1.0 - gate) * 0.5

    # --- HPWL-feedback (gentle, bounded) ---------------------------------
    if hpwl_history and len(hpwl_history) >= 6:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Plateau: progress has stalled -> sharpen slightly to refine.
            if prev > 0.0 and (prev - best_recent) / prev < 1.0e-3:
                gamma *= 0.88

            # Diverging: HPWL climbing -> smooth back out for stability.
            if last > first * 1.015:
                gamma *= 1.25
            # Healthy descent: nudge toward accuracy.
            elif last < first * 0.99:
                gamma *= 0.96

    # --- late-stage accuracy ceiling ------------------------------------
    # Keep the tail accurate, but only once density has genuinely settled.
    if progress > 0.85:
        ceil = 1.2 if ov > 0.12 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.2 if ov > 0.12 else 1.3)

    # --- final clamp -----------------------------------------------------
    if gamma != gamma:                            # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))