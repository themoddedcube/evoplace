import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for differentiable global placement.

    Primary signal is overflow (DREAMPlace-style): when cells are still
    clustered (high overflow) we want large gamma for smooth gradients; as the
    layout spreads (low overflow) gamma shrinks toward an accurate HPWL
    approximation. Progress and HPWL trend act only as gentle, bounded
    modifiers so the schedule stays monotone-ish and never destabilizes.
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

    # --- Core overflow-adaptive term (log-linear, the DREAMPlace workhorse) ---
    # ov in [0,1] -> gamma in [gamma_low, gamma_high] geometrically.
    # Geometric (log) interpolation keeps gradients well-scaled across the run.
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    # Smoothstep on overflow: be aggressive about lowering gamma only once
    # overflow has genuinely dropped, but keep it high while still congested.
    ov_s = ov * ov * (3.0 - 2.0 * ov)
    gamma_ov = math.exp(log_lo + (log_hi - log_lo) * ov_s)

    # --- Progress term: ensures decay even if overflow plateaus high ---
    # Cosine annealing in log-space from gamma_high -> gamma_low.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # Blend: overflow leads early, progress guarantees fine-tuning late.
    w_prog = progress
    gamma = (1.0 - w_prog) * gamma_ov + w_prog * min(gamma_ov, gamma_prog)

    # --- Bounded HPWL-trend feedback (small, stable nudges only) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Stagnation: nothing improving -> sharpen a touch to escape plateau.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # Mild divergence: HPWL creeping up -> smooth gradients a bit.
            if window[-1] > window[0] * 1.03:
                gamma *= 1.15
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # --- Late-stage ceilings for accurate final HPWL ---
    if progress > 0.90:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.75:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.0)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))