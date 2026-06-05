import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven exponential gamma schedule with progress modulation.

    Primary signal is overflow (the canonical DREAMPlace driver): high
    overflow => clustered cells => high smooth gamma; low overflow =>
    spread cells => low accurate gamma. A mild progress term biases the
    late phase toward fine-tuning, and a stagnation/divergence guard on
    hpwl_history nudges gamma without destabilizing the descent.
    """

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary: geometric interpolation driven by overflow ---
    # ov=1 -> gamma_high, ov=0 -> gamma_low. Self-adaptive and robust:
    # gamma naturally tracks the true clustering state of the layout.
    ratio = gamma_low / gamma_high
    ov_curve = ov ** 1.1                       # slight emphasis on relieving high overflow
    base_ov = gamma_high * (ratio ** (1.0 - ov_curve))

    # --- secondary: monotone progress decay (cosine) for fine-tuning ---
    # Blended in softly so the schedule still descends even if overflow
    # plateaus, but overflow remains the dominant driver early on.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base_prog = gamma_high * (ratio ** cos_prog)

    # Weight shifts from progress-led smoothing early to overflow-led late,
    # but never lets either term dominate completely.
    w_ov = 0.45 + 0.30 * progress
    gamma = w_ov * base_ov + (1.0 - w_ov) * base_prog

    # --- hpwl feedback: gentle, bounded adjustments ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # diverging: HPWL climbing -> smooth more to recover
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25
            # improving steadily -> sharpen for accuracy
            elif window[-1] < window[0] * 0.985:
                gamma *= 0.92
            # stagnating -> sharpen modestly to escape coarse minimum
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

    # --- late-phase ceilings: force accurate HPWL once nearly placed ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guard ---
    if gamma != gamma:  # NaN
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))