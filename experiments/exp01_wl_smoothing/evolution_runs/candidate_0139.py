import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for differentiable global placement.

    Smooth (high gamma) while density overflow is large and cells are still
    spreading; accurate (low gamma) as overflow collapses and we fine-tune.
    Progress provides a monotone decay fallback so gamma anneals even if the
    overflow signal is noisy or stalls.
    """

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                       # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = gamma_low / gamma_high             # in (0, 1)

    # --- primary driver: overflow (DREAMPlace-style log-linear mapping) ---
    # ov = 1  -> gamma_high (smooth gradients, cells clustered)
    # ov = 0  -> gamma_low  (accurate HPWL, fine-tuning)
    ov_eff = ov ** 0.85                            # mild emphasis on lowering early
    ov_term = gamma_high * (log_ratio ** (1.0 - ov_eff))

    # --- secondary driver: cosine-annealed progress (monotone decay fallback) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_term = gamma_high * (log_ratio ** cos_prog)

    # --- blend: overflow leads, progress stabilizes ---
    gamma = 0.70 * ov_term + 0.30 * prog_term

    # --- gentle history-based adaptation (plateau / divergence) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # plateau: tighten approximation to escape with accurate gradients
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # diverging: HPWL trending up -> smooth gradients to recover
            if window[-1] > window[0] * 1.03:
                gamma *= 1.25
            elif window[-1] < window[0] * 0.98:    # steady improvement
                gamma *= 0.97

    # --- late-stage accuracy caps (only when density is settling) ---
    if progress > 0.85:
        ceil = 1.2 if ov > 0.10 else gamma_low
        gamma = min(gamma, ceil)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final NaN guard + clamp ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))