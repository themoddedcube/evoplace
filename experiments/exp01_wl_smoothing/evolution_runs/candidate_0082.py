import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware annealed gamma schedule for WA-WL placement."""

    # --- robust input sanitation ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # --- primary driver: overflow-adaptive geometric interpolation ---
    # When cells are still spread out (high overflow) we want a smooth, high
    # gamma; as the layout legalizes (overflow -> 0) we anneal toward gamma_low
    # for an accurate HPWL approximation. Overflow leads, progress assists.
    ov_drive = ov ** 0.65  # emphasize the move toward low gamma as ov shrinks

    # progress floor so gamma keeps descending even if overflow stalls high
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_drive = 1.0 - cos_prog

    # blend the two drivers; weight overflow more early, progress more late
    w_prog = 0.35 + 0.30 * progress
    drive = (1.0 - w_prog) * ov_drive + w_prog * prog_drive
    drive = min(1.0, max(0.0, drive))

    # geometric (log-linear) interpolation between low and high gamma
    log_gamma = log_low + (log_high - log_low) * drive
    gamma = math.exp(log_gamma)

    # --- HPWL-history feedback (stagnation / divergence control) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0 and h != float("inf")]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # stagnating improvement -> sharpen (lower gamma) for accuracy
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # diverging / oscillating up -> smooth (raise gamma) to stabilize
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25

    # --- late-stage ceiling: force fine-tuning regime near the end ---
    if progress > 0.80:
        ceil = 1.5 if ov > 0.10 else 0.8
        gamma = min(gamma, ceil)

    # --- final NaN/inf-safe clamp to legal range ---
    if gamma != gamma or gamma == float("inf") or gamma == float("-inf"):
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))