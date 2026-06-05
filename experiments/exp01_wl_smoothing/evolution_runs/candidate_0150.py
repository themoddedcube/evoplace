import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- Base annealing: log-space cosine decay high -> low ---
    # Cosine gives a gentle hold at high gamma early (cells still clustering)
    # then accelerates the descent toward low gamma for fine HPWL accuracy.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # --- Overflow coupling ---
    # Density is the physically meaningful signal: while bins are still
    # heavily over-full we must keep gradients smooth (high gamma); once the
    # layout is nearly legal we can trust low gamma regardless of iteration.
    # Blend the schedule toward an overflow-driven target.
    ov_target = gamma_low + (gamma_high - gamma_low) * (ov ** 1.4)
    # weight on the overflow target grows as the run progresses, so early on
    # we follow the annealing curve and late we follow legality.
    w_ov = 0.30 + 0.45 * progress
    gamma = (1.0 - w_ov) * base + w_ov * ov_target

    # A mild multiplicative nudge so high overflow never sits at low gamma.
    gamma *= 0.70 + 0.55 * (ov ** 1.2)

    # --- HPWL trend feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Stalled improvement -> sharpen (lower gamma) to chase HPWL,
            # but only when density is acceptable, else we risk instability.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85 if ov < 0.12 else 0.95

            # Diverging HPWL -> back off to smoother gradients.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # --- Late-stage ceilings: commit to accurate HPWL once nearly legal ---
    if progress > 0.85:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))