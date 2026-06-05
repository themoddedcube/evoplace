def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base multiplicative growth that anneals toward 1 as iterations proceed,
    # so the density penalty ramps hard early and gently later.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while many bins are still
    # over-dense, ease off as the layout legalizes.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from history: if overflow is stalling/rising, grow faster;
    # if it is dropping steadily, relax the multiplier toward LOWER_PCOF.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            trend = last - prev  # >0 means overflow getting worse

    # Blend: high overflow or worsening trend -> closer to base (aggressive);
    # low overflow or improving trend -> closer to LOWER_PCOF (gentle).
    aggression = min(max(of + 5.0 * max(trend, 0.0), 0.0), 1.0)
    mu = LOWER_PCOF + aggression * (base - LOWER_PCOF)

    # Damp growth if gradients are exploding to keep optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1.0:
            mu = 1.0 + (mu - 1.0) / (1.0 + 0.1 * (gradient_norm - 1.0))

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda

    # Enforce required output range.
    return float(min(max(new_lambda, 0.01), 50.0))