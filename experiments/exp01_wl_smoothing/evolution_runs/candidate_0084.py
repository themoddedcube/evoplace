import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule (DREAMPlace-style) with progress
    coupling and HPWL plateau adaptation. High gamma while cells are
    spread (high overflow), annealing toward low gamma as the layout
    legalizes for accurate HPWL fine-tuning."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- primary driver: overflow-adaptive (log-linear) ---
    # DREAMPlace couples gamma to overflow: gamma large while bins are
    # over-dense, small once density settles. Geometric interpolation in
    # log-space gives smooth, well-conditioned transitions.
    # ov=1 -> gamma_high, ov=0 -> gamma_low.
    ov_shaped = ov ** 0.85          # slightly front-load: stay smooth longer
    base_ov = gamma_high * (gamma_low / gamma_high) ** (1.0 - ov_shaped)

    # --- secondary driver: schedule progress (cosine anneal) ---
    # Guarantees decay even if overflow stalls; ties late iterations to
    # accurate HPWL regardless of density readings.
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    base_prog = gamma_high * (gamma_low / gamma_high) ** cos_prog

    # Blend: overflow leads early (placement-driven), progress takes over
    # late to force fine-tuning. Weight shifts with progress.
    w_prog = progress ** 1.5
    gamma = (1.0 - w_prog) * base_ov + w_prog * min(base_ov, base_prog)

    # --- HPWL-history adaptation ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # plateau: HPWL no longer improving -> sharpen (lower gamma)
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.88

            # divergence: HPWL climbing -> re-smooth (raise gamma) to
            # recover gradient quality, but only if density justifies it.
            if window[-1] > window[0] * 1.02 and ov > 0.05:
                gamma *= 1.25

    # --- late-phase ceiling: enforce accurate-HPWL regime near the end ---
    if progress > 0.75:
        ceil = 1.5 if ov > 0.10 else 0.7
        gamma = min(gamma, ceil)

    # --- legalized layout: push toward accurate regime ---
    if ov < 0.05:
        gamma = min(gamma, gamma_low)

    return min(50.0, max(0.01, gamma))