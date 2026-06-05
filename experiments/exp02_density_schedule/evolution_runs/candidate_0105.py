def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty schedule with bounded growth."""
    # Base DREAMPlace-style geometric ramp on the penalty multiplier.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.003
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Overflow-adaptive scaling: push harder while cells are still spread out,
    # ease off as the layout legalizes so we don't overshoot and diverge.
    # Interpolate the multiplier between a gentle floor and the base ramp by overflow.
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Trend awareness: if overflow has stalled (not decreasing), nudge harder.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:
            mu *= 1.02

    # Gradient safety: if gradients blow up, damp the penalty growth.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.01)

    # Keep the per-step multiplier sane.
    mu = min(max(mu, 0.95), 1.10)

    next_lambda = current_lambda * mu
    if next_lambda != next_lambda:  # NaN guard
        next_lambda = current_lambda

    # Enforce the hard return bounds.
    return float(min(max(next_lambda, 0.01), 50.0))