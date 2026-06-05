def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight multiplier with annealing and clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Geometric ramp that decays toward a floor as iterations proceed.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Normalize overflow into [0, 1].
    of = overflow if overflow is not None else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Push hard while many bins are over-dense; ease off near legalization so
    # wirelength can be fine-tuned. Interpolate the multiplier by overflow.
    mu = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Stall detection: if overflow is not improving, nudge the penalty harder.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev is not None and of >= prev - 1e-4:
            mu *= 1.02

    # Gradient safety: soften the update if gradients explode.
    if gradient_norm is not None and gradient_norm > 1.0e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range to avoid divergence.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)