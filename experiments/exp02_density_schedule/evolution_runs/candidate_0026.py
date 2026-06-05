def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ Overflow-adaptive density-weight (lambda) multiplier with hard caps. """
    # Sanitize inputs (the previous version diverged to inf because lambda grew
    # unbounded and nothing guarded against runaway gradients).
    of = overflow if overflow is not None else 1.0
    if of != of:                      # NaN guard
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    gn = gradient_norm if (gradient_norm is not None and gradient_norm == gradient_norm) else 0.0
    cl = current_lambda if (current_lambda is not None and current_lambda == current_lambda) else 0.01

    # Gentle geometric growth, scaled by how crowded the bins still are.
    # When overflow is high -> push density weight up; as it legalizes
    # (overflow -> 0) the multiplier eases toward 1 so HPWL can settle.
    base = 1.04
    mu = 1.0 + (base - 1.0) * (of ** 0.5)

    # Stagnation kick: if overflow has plateaued while still crowded, nudge harder.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        if (max(recent) - min(recent)) < 1e-3 and of > 0.1:
            mu *= 1.015

    # Slow start so early, noisy iterations don't over-penalize density.
    if iteration < 10:
        mu = 1.0 + (mu - 1.0) * 0.5

    # Gradient safety valve: hold lambda steady when gradients blow up.
    if gn > 1e3:
        mu = 1.0

    new_lambda = cl * mu

    # Legal range, with a practical ceiling to prevent divergence.
    return float(min(max(new_lambda, 0.01), 50.0))