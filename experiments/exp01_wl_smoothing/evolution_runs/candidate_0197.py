import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven log-interpolated gamma with smooth progress decay."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_ratio = math.log(gamma_high / gamma_low)

    # --- Primary driver: overflow (proven robust in DREAMPlace/RePlAce) ---
    # High overflow (early, cells spread/clustered) -> high gamma (smooth grads).
    # Low overflow (late, legal-ish) -> low gamma (accurate HPWL).
    # Geometric (log-linear) interpolation keeps gamma within [low, high]: no blow-up.
    f_ov = ov ** 0.85          # slightly convex: hold high gamma until overflow truly drops

    # --- Secondary driver: schedule progress (monotone fine-tuning pressure) ---
    # Cosine ramp so the early phase stays smooth and the tail anneals gently.
    f_prog = 0.5 + 0.5 * math.cos(math.pi * progress)   # 1 -> 0

    # Blend: overflow dominates, progress guarantees decay even if overflow stalls.
    f = 0.70 * f_ov + 0.30 * f_prog
    f = min(1.0, max(0.0, f))

    gamma = gamma_low * math.exp(log_ratio * f)

    # --- Gentle plateau / divergence adaptation (multiplicative, bounded) ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Stagnation: sharpen toward accurate HPWL to escape flat region.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

            # Divergence: HPWL rising -> smooth gradients to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.25
            # Healthy descent: nudge sharper to lock in gains.
            elif window[-1] < window[0] * 0.97:
                gamma *= 0.93

    # --- Late-stage caps: force accurate HPWL once mostly legal ---
    if progress > 0.85:
        gamma = min(gamma, 1.2 if ov > 0.10 else 0.6)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))