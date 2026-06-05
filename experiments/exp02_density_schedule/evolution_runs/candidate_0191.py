def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth, annealing toward 1.0 as placement matures.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow (guard NaN/inf) and clamp to [0, 1].
    of = overflow if overflow == overflow else 1.0
    if of > 1.0:
        of = 1.0
    elif of < 0.0:
        of = 0.0

    # Overflow-adaptive: grow the density penalty hard while bins are congested,
    # and ease off smoothly as the layout legalizes (overflow -> 0).
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = base * adapt

    # Trend control from history: accelerate if overflow is rising (cells still
    # spreading badly), damp if it has plateaued to avoid over-penalizing.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[-1] - recent[0]
        if delta > 1e-3:
            mu *= 1.02
        elif abs(delta) < 1e-4:
            mu *= 0.99

    # Stability brake: back off growth if gradients blow up.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu *= 0.97

    new_lambda = current_lambda * mu

    # Enforce required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)