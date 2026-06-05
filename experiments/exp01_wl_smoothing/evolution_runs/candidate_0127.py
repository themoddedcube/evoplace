import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma annealing with progress safeguard and plateau response."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:               # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- primary signal: overflow drives smoothness in log-space ---
    # Overflow is the true state of the placement (high while cells overlap,
    # low once spread). Map it through a mild convex curve so gamma stays high
    # while density is bad and drops sharply only as bins clear out.
    ov_coord = ov ** 0.85
    log_gamma_ov = log_lo + (log_hi - log_lo) * ov_coord

    # --- secondary signal: iteration progress as a monotone backstop ---
    # Cosine schedule guarantees descent even if overflow stalls or is noisy.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_gamma_prog = log_hi + (log_lo - log_hi) * cos_prog

    # Blend: trust overflow more early, lean on progress to force convergence late.
    w_prog = 0.30 + 0.45 * progress
    log_gamma = (1.0 - w_prog) * log_gamma_ov + w_prog * log_gamma_prog
    gamma = math.exp(log_gamma)

    # --- plateau / divergence response from HPWL trend ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0.0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # stalled improvement -> sharpen toward accurate HPWL
            if prev > 0.0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # HPWL rising (instability) -> smooth gradients back up
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.92

    # --- late-stage caps so fine-tuning happens at accurate, low gamma ---
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    # --- final NaN/range guard ---
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))