def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty multiplier with safe clamping."""
    # Base geometric ramp (DREAMPlace-style), annealed so growth slows late.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.0
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize inputs (evolution harness may pass odd values).
    of = overflow if (overflow == overflow and overflow >= 0.0) else 1.0
    cl = current_lambda if (current_lambda == current_lambda and current_lambda > 0.0) else 0.01

    # Overflow-adaptive scaling: push lambda harder while cells are still
    # spread out (high overflow), ease off as the layout becomes legal so we
    # don't over-penalize density at the expense of wirelength.
    # Map overflow in [~0.05, 1.0] -> multiplier in [LOWER_PCOF, UPPER_PCOF].
    target_of = 0.10
    if of > target_of:
        adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * min((of - target_of) / (1.0 - target_of), 1.0)
    else:
        # Below target: gently relax the penalty to recover wirelength.
        adapt = LOWER_PCOF - 0.02 * (1.0 - of / target_of)

    # Trend awareness: if overflow has stalled (not decreasing), nudge harder.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last and last >= prev - 1e-4 and of > target_of:
            adapt *= 1.02  # stalled while still illegal -> stronger push

    mu = 0.5 * base_mu + 0.5 * adapt

    # Damp huge density gradients to avoid blow-ups.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = cl * mu

    # Hard clamp to the required range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)