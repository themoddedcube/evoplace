import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-decayed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:            # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base schedule: smooth log-cosine decay high -> low ---
    # cos_prog goes 0 -> 1 with an ease-in/ease-out profile, so gamma stays
    # high while cells are still clustering and drops late for fine HPWL.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- overflow coupling ---
    # When overflow is high the layout is still legalizing, so keep gamma
    # large (smooth gradients); as bins drain, trust the schedule's low end.
    ov_floor = 0.45 + 0.55 * (ov ** 1.3)        # multiplicative pull toward base
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    gamma = 0.6 * base * ov_floor + 0.4 * ov_target

    # --- HPWL feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first, last = window[0], window[-1]

            # Plateau: relative improvement stalled -> sharpen toward true HPWL.
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (HPWL climbing) -> smooth gradients to recover.
            if first > 0.0 and last > first * 1.02:
                gamma *= 1.35
            # Healthy descent -> nudge a touch sharper.
            elif first > 0.0 and last < first * 0.98:
                gamma *= 0.93

    # --- late-stage ceilings: force accurate HPWL near convergence ---
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.65)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.4)

    # --- final NaN/range guard ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))