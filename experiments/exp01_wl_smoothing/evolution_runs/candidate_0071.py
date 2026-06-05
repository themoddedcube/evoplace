import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-aware cosine-annealed gamma schedule with plateau/divergence adaptation."""

    # --- sanitize inputs ---
    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    if progress != progress:  # NaN guard
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # --- base decay: cosine annealing in log-space ---
    # Smoothly transitions high -> low; flatter at the ends, steepest in the middle,
    # which keeps gradients smooth early and HPWL accurate late.
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    base = math.exp(log_lo + (log_hi - log_lo) * cos_factor)

    # --- overflow coupling ---
    # When density overflow is high, cells are still poorly spread: keep gamma high
    # for smoother gradients. As overflow collapses, let gamma fall toward gamma_low.
    overflow_factor = 0.55 + 1.9 * (ov ** 1.3)
    gamma = base * overflow_factor

    # --- HPWL-history feedback ---
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-6:] if h is not None and h == h]
        if len(recent) >= 5:
            window = recent[-5:]
            prev = recent[-6] if len(recent) >= 6 else window[0]
            best_recent = min(window)
            first = window[0]
            last = window[-1]

            # Plateau: relative improvement stalled -> sharpen approximation (lower gamma)
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.75

            # Divergence: HPWL climbing -> back off to smoother gradients (raise gamma)
            if first > 0 and last > first * 1.02:
                gamma *= 1.4

    # --- late-stage clamp: prioritize accurate HPWL near convergence ---
    if progress > 0.9:
        gamma = min(gamma, 0.9)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if gamma != gamma:  # final NaN guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))