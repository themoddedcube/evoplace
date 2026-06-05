def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base DREAMPlace-style multiplicative ramp on the density penalty.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction: push harder while bins are congested,
    # relax once the layout spreads out so HPWL can be fine-tuned.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from recent overflow history (negative => improving).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = [h for h in overflow_history[-5:] if h == h]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]

    if of > 0.10:
        # Still congested: accelerate the penalty, more so if not improving.
        mu = base_mu * (1.0 + 0.5 * (of - 0.10))
        if trend > 0.0:
            mu *= 1.05
    else:
        # Nearly legal: ease off to let wirelength settle.
        mu = max(base_mu * (1.0 - (0.10 - of)), LOWER_PCOF)

    # Gradient-norm safeguard: damp updates when gradients explode.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 0.0 and gn > 1e3:
        mu = min(mu, UPPER_PCOF)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN fallback
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))