import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware geometric annealing with plateau adaptation."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Geometric (log-space) decay along a cosine-shaped progress curve:
    # holds gamma high a bit longer early, then drops smoothly into fine-tuning.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Overflow coupling. While cells are still spread (high overflow) we want
    # smoother gradients, so pull gamma up; as the layout legalizes (overflow
    # -> 0) the additive/multiplicative overflow terms relax toward gamma_low.
    ov_mult = 0.5 + 1.7 * (ov ** 1.3)
    ov_add = gamma_low + (gamma_high - gamma_low) * (ov ** 1.6)

    # Weight the overflow-driven term more in the early/clustering phase and let
    # the schedule term dominate late, so fine-tuning is not held back by a
    # lingering density signal.
    w_sched = 0.45 + 0.25 * progress
    gamma = w_sched * base * ov_mult + (1.0 - w_sched) * ov_add

    # HPWL-history feedback: react to plateaus and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stalled improvement: sharpen to chase accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80

            # Diverging (recent worse than start of window): smooth out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.35
            # Healthy descent: hold steady, nudge slightly sharper.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.93

    # Late-stage ceilings force accurate HPWL once the layout is essentially
    # legal; loosen the ceiling if overflow is still non-trivial.
    if progress > 0.90:
        ceil = 1.3 if ov > 0.10 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.80:
        gamma = min(gamma, 2.2 if ov > 0.10 else 1.2)
    elif progress > 0.65:
        gamma = min(gamma, 3.0 if ov > 0.10 else 2.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))