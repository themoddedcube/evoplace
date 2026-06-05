import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with progress floor and plateau control."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Primary driver: overflow ---
    # While bins are densely over-packed (cells still clustered/spreading),
    # keep gamma high for smooth gradients. As overflow drops, anneal toward
    # an accurate, low-gamma approximation. Geometric interpolation in log
    # space gives a perceptually smooth ramp across the wide [0.5, 8.0] range.
    # Shape the overflow with a mild power so most of the annealing happens
    # in the low-overflow regime where fine placement actually matters.
    ov_shaped = ov ** 0.85
    ov_base = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_shaped)

    # --- Secondary driver: progress (cosine annealing) ---
    # Provides a monotone time-based pull toward low gamma so the schedule
    # still converges even if overflow stalls at a moderate value.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: early on trust overflow, late lean on the time schedule.
    w_prog = 0.30 + 0.45 * progress
    gamma = (1.0 - w_prog) * ov_base + w_prog * prog_base

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: HPWL barely improving -> sharpen (lower gamma) to
            # recover approximation accuracy and escape the flat region.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging: HPWL climbing -> smooth gradients again.
            if last > first * 1.02:
                gamma *= 1.30
            # Improving steadily: nudge sharper to lock in accurate WL.
            elif last < first * 0.985:
                gamma *= 0.92

    # --- End-game ceilings: force accurate WL once nearly placed ---
    if progress > 0.90:
        ceil = 1.2 if ov > 0.08 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.78:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.3)
    elif progress > 0.60:
        gamma = min(gamma, 4.0 if ov > 0.15 else 2.5)

    # Floor relative to overflow: never go ultra-sharp while badly overflowed.
    if ov > 0.5:
        gamma = max(gamma, 2.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))