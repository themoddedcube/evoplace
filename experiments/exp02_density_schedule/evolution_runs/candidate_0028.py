def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative growth that anneals down as iterations proceed,
    # so early pressure is strong and late updates are gentle.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1].
    of = overflow
    if of != of or of is None:  # NaN / None guard
        of = 1.0
    of = min(max(float(of), 0.0), 1.0)

    # Overflow trend: positive/zero means overflow has stalled or grown.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            trend = last - prev

    # Overflow-adaptive blend: push lambda harder while many bins are
    # overfull, ease off as the placement legalizes.
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = 0.5 * base_mu + 0.5 * adapt

    # If overflow is stalling at a high level, add a little extra pressure.
    if trend >= 0.0 and of > 0.10:
        mu *= 1.01

    # Near-legal: stop growing lambda and lock in wirelength.
    if of < 0.05:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Hard clamp to the allowed range (prevents divergence to inf).
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return new_lambda