def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """Overflow-adaptive density-penalty growth with hard clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Base DREAMPlace-style decaying growth factor in [LOWER_PCOF, UPPER_PCOF].
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while bins are congested,
    # ease off (toward 1.0 = no growth) as the layout legalizes.
    of = overflow if overflow == overflow else 1.0          # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from history: if overflow is stalling/rising, grow a touch faster.
    trend = 0.0
    if overflow_history:
        prev = overflow_history[-1]
        if prev == prev:
            trend = of - prev                                # >0 means worsening

    mu = 1.0 + (base - 1.0) * (0.25 + 0.75 * of) + 0.5 * max(trend, 0.0)
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    # Gradient-norm safeguard: if gradients explode, stop amplifying lambda.
    if gradient_norm == gradient_norm and gradient_norm > 1e6:
        mu = LOWER_PCOF

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:                             # NaN fallback
        new_lambda = current_lambda

    # Enforce the required output range.
    return float(min(max(new_lambda, 0.01), 50.0))