def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base subgradient growth that cools as iterations progress
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push harder while bins are congested,
    # ease off as the layout spreads out so HPWL can be fine-tuned.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    if of > 0.10:
        # Still congested: accelerate density penalty growth.
        mu = base_mu * (1.0 + 0.5 * of)
    else:
        # Nearly legal: relax toward 1.0 to stop over-driving the penalty.
        relax = max(LOWER_PCOF, 1.0 + (base_mu - 1.0) * (of / 0.10))
        mu = relax

    # Detect overflow stagnation/divergence from history and damp growth.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:  # not decreasing
            mu = min(mu, 1.0 + 0.3 * (mu - 1.0)) if mu > 1.0 else mu

    # Gradient-norm safety: if gradients explode, avoid amplifying lambda.
    if gradient_norm == gradient_norm and gradient_norm > 1e6:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to the required valid range.
    if new_lambda != new_lambda:  # NaN fallback
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))