import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for WA-wirelength smoothing.

    Primary signal is overflow (true placement state): high overflow ->
    high gamma (smooth gradients, cells still clustering); low overflow ->
    low gamma (accurate HPWL, fine-tuning). Progress acts as a backstop so
    gamma keeps annealing even if overflow stalls high; HPWL trend supplies
    divergence recovery and plateau sharpening.
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

    # Overflow is the dominant driver: log-linear interpolation in overflow.
    # ov=1 -> gamma_high, ov=0 -> gamma_low.
    ov_term = gamma_low * (gamma_high / gamma_low) ** (ov ** 0.85)

    # Progress backstop: cosine annealing in log space so gamma still relaxes
    # smoothly toward gamma_low even when overflow plateaus.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_term = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Weight toward overflow (the real convergence signal), keep progress as a guard.
    gamma = 0.7 * ov_term + 0.3 * prog_term

    # HPWL-trend response (ignore None / NaN / inf entries).
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0.0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Diverging wirelength -> add smoothing to recover stability.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Steady improvement -> sharpen gamma for finer HPWL accuracy.
            elif window[-1] < window[0] * 0.995:
                gamma *= 0.92

            # Hard plateau -> nudge gamma down to escape the smooth-but-loose regime.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # Late-stage ceilings: enforce accurate gamma once mostly legalized.
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.2)

    if gamma != gamma or gamma == float("inf"):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))