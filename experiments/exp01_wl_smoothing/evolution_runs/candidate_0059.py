import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-adaptive cosine-annealed gamma schedule."""

    total = total_iterations if total_iterations and total_iterations > 0 else 1
    progress = iteration / total
    progress = min(1.0, max(0.0, progress))
    ov = overflow if overflow is not None else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Cosine annealing in log-space between high and low gamma.
    log_hi = math.log(gamma_high)
    log_lo = math.log(gamma_low)
    cos_frac = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    base = math.exp(log_lo + (log_hi - log_lo) * cos_frac)

    # Overflow drives smoothness: spread cells (high gamma) while bins are
    # over-dense, relax toward accurate HPWL as the layout legalizes.
    overflow_factor = 0.5 + 2.5 * (ov ** 1.3)
    gamma = base * overflow_factor

    # Couple gamma to overflow level directly so a stalled-but-dense layout
    # keeps enough smoothing to keep moving cells apart.
    if ov > 0.85:
        gamma = max(gamma, 4.0)
    elif ov < 0.10:
        gamma = min(gamma, 1.5)

    # HPWL-history feedback: react to plateaus and divergence.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = hpwl_history[-5:]
        prev = hpwl_history[-6] if len(hpwl_history) >= 6 else recent[0]
        best_recent = min(recent)
        finite = [h for h in recent if h == h and abs(h) != float("inf")]

        if not finite:
            gamma *= 1.5  # recover from blown-up / nan HPWL with smoother grads
        else:
            # Plateau: nudge gamma down to sharpen the HPWL approximation,
            # but only once overflow is low enough that it is safe to do so.
            if prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.8 if ov < 0.3 else 0.95
            # Divergence: HPWL climbing -> add smoothing to stabilize.
            if recent[0] > 0 and recent[-1] > recent[0] * 1.02:
                gamma *= 1.4

    # Late-stage fine-tuning: force low gamma for accurate wirelength.
    if progress > 0.9:
        gamma = min(gamma, 0.8)
    elif progress > 0.8:
        gamma = min(gamma, 1.5)

    if gamma != gamma:  # nan guard
        gamma = gamma_low

    return min(50.0, max(0.01, gamma))