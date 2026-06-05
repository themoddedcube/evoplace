import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-interpolated gamma with gentle progress decay."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:                      # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- primary driver: overflow ---
    # Overflow is the true physical indicator of placement state. While cells
    # are still spread out (high overflow) we want smooth gradients (high gamma);
    # as the layout legalizes (overflow -> 0) we sharpen toward accurate HPWL.
    # Interpolate in log-space so the geometric scale is respected.
    ov_t = ov ** 0.85                              # mild concavity: stay smooth longer
    log_ov = log_lo + (log_hi - log_lo) * ov_t

    # --- secondary driver: schedule progress (cosine in log-space) ---
    # Guarantees monotone-ish annealing even if overflow plateaus, and provides
    # a floor of decay late in the run so we always fine-tune.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_prog = log_hi + (log_lo - log_hi) * cos_prog

    # Blend: overflow leads early, the progress schedule asserts itself late so
    # we never get stuck at high gamma if overflow is slow to clear.
    w_prog = 0.35 + 0.45 * progress                # 0.35 -> 0.80
    log_gamma = (1.0 - w_prog) * log_ov + w_prog * log_prog
    gamma = math.exp(log_gamma)

    # --- HPWL-history feedback (gentle, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # Stagnation: progress has stalled -> sharpen slightly to refine.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.90

            # Divergence: HPWL climbing -> smooth gradients to restabilize.
            if window[-1] > window[0] * 1.03:
                gamma *= 1.25
            elif window[-1] < window[0] * 0.97:
                # Healthy descent -> let sharpening continue.
                gamma *= 0.97

    # --- late-stage caps to lock in an accurate HPWL ---
    if progress > 0.90:
        ceil = 1.2 if ov > 0.08 else 0.6
        gamma = min(gamma, ceil)
    elif progress > 0.75:
        gamma = min(gamma, 2.5 if ov > 0.08 else 1.3)

    # Keep gamma from collapsing too early while cells are still mobile,
    # which is the usual cause of divergence (noisy gradients -> inf HPWL).
    if progress < 0.5 and ov > 0.5:
        gamma = max(gamma, 2.0)

    if gamma != gamma:                             # final NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))