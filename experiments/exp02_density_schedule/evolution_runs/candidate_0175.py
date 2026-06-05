def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Iteration-decayed base growth (matches original spirit: strong early, gentle late)
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow severity: push density force harder while many bins are over-filled,
    # ease off as the layout legalizes so wirelength can be fine-tuned.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    severity = 1.0 + 0.4 * (of - 0.1)                       # >1 when congested, <1 when sparse

    # Overflow trend: if overflow is stalling/rising, accelerate; if falling fast, relax.
    trend = 0.0
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:                                    # guard NaN
            trend = of - prev
    if trend >= 0.0 and of > 0.1:
        severity *= 1.05                                    # stuck congested -> stronger ramp
    elif trend < -0.02:
        severity *= 0.97                                    # legalizing well -> coast

    mu = base * severity

    # Gradient-norm safety: damp the ramp if gradients are exploding to avoid divergence.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, LOWER_PCOF * 1.05)
    else:
        mu = 1.0

    # Keep mu in a sane multiplicative band, then update and hard-clamp lambda.
    mu = min(max(mu, LOWER_PCOF), 1.10)
    new_lambda = current_lambda * mu

    if new_lambda != new_lambda:                            # final NaN guard
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))