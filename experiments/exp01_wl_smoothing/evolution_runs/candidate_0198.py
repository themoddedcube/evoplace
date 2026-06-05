import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with cosine guide and history feedback."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- overflow-driven core (DREAMPlace-style, log-linear) ---
    # High overflow => smooth gradients (high gamma); settled cells => accurate (low gamma).
    # ov_shaped keeps gamma high while cells are spread, then drops sharply as they settle.
    ov_shaped = ov ** 0.85
    gamma_ov = math.exp(log_lo + (log_hi - log_lo) * ov_shaped)

    # --- progress-driven cosine guide (stabilizes when overflow is noisy/stuck) ---
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    gamma_prog = math.exp(log_hi + (log_lo - log_hi) * cos_prog)

    # Trust overflow early, progress late.
    w_ov = 0.70 - 0.40 * progress
    gamma = w_ov * gamma_ov + (1.0 - w_ov) * gamma_prog

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Stagnation: negligible improvement -> sharpen for accuracy.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85
            # Divergence: HPWL climbing -> smooth out to recover.
            if last > first * 1.02:
                gamma *= 1.30
            elif last < first * 0.985:
                gamma *= 0.93

    # --- late-stage fine-tuning clamp ---
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))