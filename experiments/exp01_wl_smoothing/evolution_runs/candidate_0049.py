import math


def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule for WA-WL placement."""

    # --- sanitize inputs ---
    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base annealing: cosine in log-space ---
    # Cosine gives a gentle high-gamma plateau early (cells still spreading)
    # and a soft landing into the low-gamma fine-tuning regime late.
    cos_t = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_hi, log_lo = math.log(gamma_high), math.log(gamma_low)
    base = math.exp(log_lo + (log_hi - log_lo) * cos_t)

    # --- overflow adaptation ---
    # While many bins are still over-dense the placement is far from legal,
    # so favor smoother (higher) gamma; relax the multiplier as overflow drops.
    overflow_factor = 0.55 + 1.6 * (ov ** 1.2)
    gamma = base * overflow_factor

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)

            # Plateau: HPWL stopped improving -> sharpen toward true HPWL.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.65

            # Divergence: HPWL climbing -> smooth gradients to recover.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.5

    # --- late-stage cap: commit to accurate HPWL near the end ---
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))