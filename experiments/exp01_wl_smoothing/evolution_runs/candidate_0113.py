import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress-based decay floor.

    Primary control is overflow (as in DREAMPlace ePlace): gamma stays high
    while density overflow is large (cells still spreading) and anneals toward
    a small value as the layout settles. A progress term guarantees the
    schedule keeps cooling even if overflow stalls, and a gentle plateau
    detector trims gamma when HPWL stops improving. The schedule is kept
    monotone-friendly (no upward spikes) to avoid gradient blow-ups.
    """

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_low / gamma_high)  # negative

    # --- Overflow-driven component (dominant) ---------------------------
    # As overflow falls from 1 -> 0, gamma anneals high -> low in log space.
    # A mild convexity (ov**0.85) keeps gamma high a touch longer while the
    # layout is still congested, then drops quickly once it opens up.
    ov_shaped = ov ** 0.85
    gamma_ov = gamma_high * math.exp(log_ratio * (1.0 - ov_shaped))

    # --- Progress-driven component (cooling floor) ----------------------
    # Cosine annealing in log space: smooth high -> low independent of ov,
    # so the schedule always cools even if overflow plateaus.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = gamma_high * math.exp(log_ratio * cos_prog)

    # Blend: lean on overflow early (spreading phase), on progress later
    # (fine-tuning phase) so we never get stuck hot near the end.
    w = progress
    gamma = (1.0 - w) * gamma_ov + w * min(gamma_ov, gamma_prog)

    # --- Plateau / divergence response (downward only) ------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Improvement has stalled -> sharpen (lower gamma) for accuracy.
            if first > 0 and (first - best_recent) / first < 1e-3:
                gamma *= 0.85

            # HPWL creeping up -> already too noisy/sharp; do NOT raise gamma
            # (raising it here historically caused oscillation/divergence).
            # Instead hold steady by clamping to current value.
            if last > first * 1.01:
                gamma = min(gamma, gamma)  # no-op guard; keep monotone cooling

    # --- Late-stage ceilings to force accurate HPWL ---------------------
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))