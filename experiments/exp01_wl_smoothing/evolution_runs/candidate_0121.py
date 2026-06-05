import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Evolved gamma schedule for differentiable global placement.

    Strategy: keep gamma high while the design is still spread out
    (high overflow / early progress) for smooth, well-behaved gradients,
    then anneal toward a low gamma for an accurate HPWL approximation as
    cells settle. Overflow is the primary driver; iteration progress is a
    secondary guard so we always converge even if overflow plateaus.
    """

    # --- sanitize inputs -------------------------------------------------
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:          # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5
    log_high = math.log(gamma_high)
    log_low = math.log(gamma_low)

    # --- overflow-driven base (geometric interpolation) ------------------
    # ov near 1 -> gamma_high, ov near 0 -> gamma_low. The exponent shapes
    # the curve so gamma stays high until overflow has dropped meaningfully,
    # then falls off quickly for the fine-tuning phase.
    ov_shape = ov ** 0.85
    base_ov = math.exp(log_low + (log_high - log_low) * ov_shape)

    # --- progress-driven anneal (cosine in log-space) --------------------
    # Guarantees monotone-ish convergence even if overflow is sticky.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base_prog = math.exp(log_high + (log_low - log_high) * cos_prog)

    # Blend: early on trust progress less; weight shifts toward the
    # progress anneal in the back half to force settling.
    w_prog = 0.30 + 0.40 * progress
    gamma = (1.0 - w_prog) * base_ov + w_prog * base_prog

    # --- HPWL-history feedback ------------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # Stagnation: tighten gamma to sharpen the HPWL estimate.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

            # Diverging HPWL: gradients too noisy -> smooth more.
            if first > 0 and last > first * 1.02:
                gamma *= 1.30
            # Healthy descent: gently push toward accuracy.
            elif first > 0 and last < first * 0.97:
                gamma *= 0.93

    # --- end-game ceilings ----------------------------------------------
    if progress > 0.85:
        gamma = min(gamma, 1.5 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.5)

    # --- final guards ----------------------------------------------------
    if gamma != gamma:                # NaN guard
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))