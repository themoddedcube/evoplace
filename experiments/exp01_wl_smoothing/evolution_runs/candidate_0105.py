import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware log-cosine gamma annealing for differentiable placement.

    High gamma early (smooth gradients, clustered cells) decaying to low gamma
    late (accurate HPWL, fine-tuning), with the descent gated primarily by the
    actual density overflow rather than raw iteration count, plus a light
    plateau/divergence response from the HPWL trajectory.
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
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)

    # --- base schedule: cosine-eased decay in log space ------------------
    # cos_prog goes 0 -> 1 smoothly with progress; gentle start, gentle end.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)

    # --- overflow drives the effective annealing position ----------------
    # While the layout is still congested (high overflow) we must stay smooth,
    # regardless of how many iterations have elapsed. As overflow falls the
    # placement is essentially legal and we can afford an accurate (low) gamma.
    # Blend a time term with an overflow term; overflow dominates.
    ov_pos = 1.0 - ov ** 0.8          # 0 when fully congested, ->1 as it clears
    eff = 0.30 * cos_prog + 0.70 * ov_pos
    eff = min(1.0, max(0.0, eff))

    # geometric interpolation high -> low
    gamma = math.exp(log_hi + (log_lo - log_hi) * eff)

    # --- HPWL-trajectory feedback ---------------------------------------
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-8:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            first, last = window[0], window[-1]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else first

            # diverging / oscillating up -> back off (smoother gradients)
            if last > first * 1.02:
                gamma *= 1.30
            # plateaued: little improvement over the window -> sharpen
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.80
            # healthy descent -> nudge toward accuracy
            elif last < first * 0.98:
                gamma *= 0.92

    # --- late-stage accuracy ceiling ------------------------------------
    # Near the end, force gamma low for an accurate HPWL objective, but allow
    # a little more headroom if the layout is still not legal.
    if progress > 0.85:
        gamma = min(gamma, 1.4 if ov > 0.10 else 0.7)
    elif progress > 0.70:
        gamma = min(gamma, 2.5 if ov > 0.10 else 1.3)

    # --- final guards ----------------------------------------------------
    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))