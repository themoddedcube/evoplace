def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003

    # Base geometric growth (DREAMPlace-style), decaying with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: when overflow is high the layout is still
    # congested, so push the density weight up faster; when overflow is low
    # we are close to legal, so grow gently to avoid wirelength blow-up.
    ov = overflow if overflow == overflow else 1.0  # guard against NaN
    ov = min(max(ov, 0.0), 1.0)
    overflow_gain = 1.0 + 0.30 * (ov - 0.10)

    # Plateau detection: if overflow has stalled (not decreasing), accelerate
    # to escape the stagnant region.
    plateau_boost = 1.0
    if isinstance(overflow_history, list) and len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 0.005:
            plateau_boost = 1.08
        elif improvement > 0.05:
            # Fast progress: ease off so we don't overshoot.
            plateau_boost = 0.97

    # Gradient safeguard: if gradients explode, dampen growth for stability.
    grad_damp = 1.0
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            grad_damp = 0.95

    mu = base_mu * overflow_gain * plateau_boost * grad_damp

    # Keep the per-step multiplier in a sane band.
    mu = min(max(mu, LOWER_PCOF), 1.15)

    new_lambda = current_lambda * mu

    # Clamp output to the required range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)