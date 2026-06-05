def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with safe clamping."""
    # Base multiplicative growth (DREAMPlace-style), annealed over iterations.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.005
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while bins are congested,
    # ease off (toward 1.0) as the layout legalizes so HPWL can refine.
    of = overflow if overflow == overflow else 1.0          # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend: are we still making density progress?
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:
            trend = of - min(max(prev, 0.0), 1.0)           # >0 means worsening

    if of > 0.20:
        # High overflow: scale growth with congestion, accelerate if worsening.
        mu = base_mu * (1.0 + 0.5 * of) * (1.10 if trend > 0.0 else 1.0)
    elif of > 0.08:
        # Mid overflow: gentle, steady tightening.
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (of / 0.20)
    else:
        # Near-legal: stop inflating lambda; let wirelength gradients dominate.
        mu = 1.0

    # Gradient-norm safeguard: if gradients explode, damp the update.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e6:
        mu = min(mu, 1.0)

    cur = current_lambda if (current_lambda == current_lambda and current_lambda > 0.0) else 1.0
    new_lambda = cur * mu

    # Hard clamp to the required return range.
    if new_lambda != new_lambda:          # NaN fallback
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))