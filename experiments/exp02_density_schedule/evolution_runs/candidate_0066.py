def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Annealed base growth: aggressive early, gentler as iterations accumulate.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)
    mu = base

    # Overflow-adaptive trend: react to how density is actually evolving.
    if overflow_history and len(overflow_history) >= 2:
        prev = float(overflow_history[-2])
        delta = overflow - prev  # < 0 means overflow is dropping (good)
        if delta < 0.0:
            # Cells are spreading well: ease growth so HPWL can settle.
            slow = max(LOWER_PCOF ** min(-delta * 25.0, 4.0), 0.9)
            mu = min(base * slow, base)
        else:
            # Overflow stalled or rising: push the density penalty harder.
            mu = base * min(1.0 + delta * 25.0, UPPER_PCOF)

    # Near convergence (low overflow): throttle growth for fine HPWL tuning.
    if overflow < 0.10:
        ramp = 1.0 + (UPPER_PCOF - 1.0) * (overflow / 0.10)
        mu = min(mu, ramp)

    # Damp updates when gradients are noisy/large to avoid overshoot.
    if gradient_norm > 0.0 and gradient_norm != gradient_norm:  # NaN guard
        mu = 1.0

    new_lambda = current_lambda * mu

    # Hard clamp to the required range (prevents the blow-up that caused inf).
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)