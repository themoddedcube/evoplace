import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Log-space cosine annealing: smooth high->low over progress.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow drives gamma: while cells are still overlapping (high ov) we
    # keep gradients smooth; as the layout legalizes (ov -> 0) we sharpen.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.5)

    # Blend the time-based schedule with the overflow-driven target. Early on
    # lean on the schedule; late, when overflow is the real signal, lean on ov.
    w_ov = 0.30 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # Couple: when overflow is very low we are essentially fine-tuning HPWL,
    # so pull gamma toward the accurate (low) regime regardless of schedule.
    if ov < 0.10:
        gamma = min(gamma, gamma_low + 1.5 * ov / 0.10)

    # HPWL trend adaptation.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: relative improvement stalled -> sharpen to escape.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging (HPWL climbing): smooth gradients back out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.95

    # Late-stage ceilings to guarantee accurate HPWL near the end.
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))