import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing for differentiable placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base log-space cosine anneal (smooth high->low) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow drives the schedule more than the clock ---
    # While bins are congested, keep gradients smooth (raise gamma);
    # once spread out, trust the sharper (low-gamma) HPWL approximation.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # Blend: early iterations lean on the clock, late ones on real overflow.
    w_ov = min(1.0, 0.30 + 0.70 * progress)
    gamma = (1.0 - w_ov) * base + w_ov * (0.6 * base + 0.4 * ov_target)

    # multiplicative congestion boost (mild, never explosive)
    gamma *= 0.70 + 0.85 * (ov ** 1.1)

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # plateau: improvement stalled -> sharpen to refine wirelength
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # diverging (HPWL rising): re-smooth gradients
            if last > first * 1.02:
                gamma *= 1.40
            # improving fast: keep sharpening gently
            elif last < first * 0.97:
                gamma *= 0.92

    # --- late-stage ceilings: commit to accurate HPWL near the end ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.5)
    elif progress > 0.80:
        gamma = min(gamma, 1.8 if ov > 0.10 else 0.9)
    elif progress > 0.65:
        gamma = min(gamma, 2.8 if ov > 0.10 else 1.6)

    # --- final guards ---
    if gamma != gamma or gamma in (float("inf"), float("-inf")):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))