import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule for differentiable global placement.

    Primary signal is overflow (the true placement state): high overflow -> high
    gamma (smooth gradients while cells are still clustered), low overflow -> low
    gamma (accurate HPWL for fine-tuning). Progress supplies a monotone cosine
    anneal so gamma keeps drifting down even if overflow plateaus, and HPWL
    history damps stagnation / divergence near convergence.
    """

    gamma_high = 8.0
    gamma_low = 0.5

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    # --- overflow-driven log-linear core (DREAMPlace-style) ---
    # Map overflow in [ov_lo, ov_hi] to gamma in [gamma_low, gamma_high] on a
    # log scale; saturate outside that band.
    ov_lo, ov_hi = 0.10, 1.00
    t = (ov - ov_lo) / (ov_hi - ov_lo)
    t = min(1.0, max(0.0, t))
    log_low, log_high = math.log(gamma_low), math.log(gamma_high)
    gamma = math.exp(log_low + t * (log_high - log_low))

    # --- progress annealing ---
    # Blend in a cosine decay so that even if overflow stalls, gamma still
    # drifts lower over time to sharpen the HPWL approximation. The anneal can
    # only pull gamma down (min), never push it back up.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    prog_gamma = gamma_high * (gamma_low / gamma_high) ** cos_prog
    w = 0.30 + 0.40 * progress  # trust overflow early, progress late
    gamma = (1.0 - w) * gamma + w * min(gamma, prog_gamma)

    # --- HPWL-history adaptive damping ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best = min(window)
            prev_best = min(recent[:-1]) if len(recent) > 5 else first

            # stagnation: tighten gamma to chase a sharper objective
            if prev_best > 0 and (prev_best - best) / prev_best < 1e-3:
                gamma *= 0.85
            # divergence: HPWL rising -> smooth gradients to recover
            if last > first * 1.02:
                gamma *= 1.30
            # steady improvement: gently sharpen
            elif last < first * 0.98:
                gamma *= 0.95

    # --- late-stage ceilings to lock in accurate HPWL ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))