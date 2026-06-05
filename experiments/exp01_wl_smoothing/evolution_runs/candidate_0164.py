import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-anchored gamma schedule with cosine base decay and gentle plateau adaptation."""

    gamma_high = 8.0
    gamma_low = 0.5

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- base schedule: smooth cosine-in-log-space decay (high -> low) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow anchoring ---
    # Overflow is the most reliable physical signal of how "spread out" the
    # placement still is. While cells remain badly overlapping (high ov) we
    # want smoother gradients; once density resolves we trust HPWL accuracy.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)

    # Blend the time-based base with the overflow-based target. Early on we
    # lean on the schedule; later we let measured overflow dominate so we do
    # not over-smooth a placement that has already legalized.
    w_ov = 0.35 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # --- HPWL feedback (gentle, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Plateau: little improvement over the window -> sharpen toward
            # accurate HPWL to escape, but only modestly.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

            # Divergence: cost climbing -> back off to smoother gradients.
            if window[-1] > window[0] * 1.01:
                gamma *= 1.25
            # Healthy descent -> nudge slightly sharper for finer placement.
            elif window[-1] < window[0] * 0.985:
                gamma *= 0.93

    # --- late-stage accuracy ceilings (let HPWL approximation tighten) ---
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)

    # --- final guards ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))