import math

def gamma_schedule(
    iteration: int,
    total_iterations: int,
    overflow: float,
    hpwl_history: list,
) -> float:
    """Overflow-driven gamma schedule with progress annealing and stability guards."""

    total = total_iterations if (total_iterations and total_iterations > 0) else 1
    progress = iteration / total
    if progress != progress:
        progress = 0.0
    progress = min(1.0, max(0.0, progress))

    ov = overflow if (overflow is not None and overflow == overflow) else 1.0
    ov = min(1.0, max(0.0, ov))

    gamma_high = 8.0
    gamma_low = 0.5

    # Primary driver: DREAMPlace-style overflow-adaptive exponential.
    # overflow ~ 1 (cells spread, bins full) -> high gamma (smooth gradients)
    # overflow ~ 0 (legalized layout)        -> low gamma  (accurate HPWL)
    ov_shaped = ov ** 0.85
    log_gamma_ov = math.log(gamma_low) + ov_shaped * (math.log(gamma_high) - math.log(gamma_low))

    # Secondary driver: cosine anneal in log-space, guarantees decay even if
    # overflow stalls high (prevents the placer getting stuck at smooth/inaccurate gamma).
    cos_prog = 0.5 - 0.5 * math.cos(math.pi * progress)
    log_gamma_prog = math.log(gamma_high) + cos_prog * (math.log(gamma_low) - math.log(gamma_high))

    # Trust overflow more early; lean on progress to force convergence late.
    w = 0.70 - 0.30 * progress
    gamma = math.exp(w * log_gamma_ov + (1.0 - w) * log_gamma_prog)

    # HPWL-history feedback: react to divergence vs. plateau.
    if hpwl_history and len(hpwl_history) >= 5:
        recent = [h for h in hpwl_history[-7:]
                  if h is not None and h == h and h > 0]
        if len(recent) >= 5:
            window = recent[-5:]
            best_recent = min(window)
            prev = recent[-6] if len(recent) >= 6 else window[0]

            # HPWL climbing -> gradients too noisy; smooth them out.
            if window[-1] > window[0] * 1.02:
                gamma *= 1.30
            # Improving steadily -> let the overflow/progress drive stand.
            elif window[-1] < window[0] * 0.98:
                gamma *= 0.97
            # Flat plateau with no net improvement -> sharpen to refine wirelength.
            elif prev > 0 and (prev - best_recent) / prev < 1e-3:
                gamma *= 0.85

    # Late-stage ceilings: guarantee accurate HPWL approximation at the finish,
    # but stay a touch higher while overflow is still non-trivial.
    if progress > 0.90:
        gamma = min(gamma, 1.0 if ov > 0.10 else 0.5)
    elif progress > 0.75:
        gamma = min(gamma, 2.0 if ov > 0.10 else 1.2)

    if gamma != gamma:
        gamma = gamma_low
    return min(50.0, max(0.01, gamma))