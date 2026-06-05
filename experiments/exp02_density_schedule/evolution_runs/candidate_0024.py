def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative density-weight growth, but made
    # overflow-adaptive and bounded so lambda never explodes to inf.

    # Base geometric ramp: strong early push, decaying toward 1.0.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push hard while bins are congested,
    # ease off (mu -> ~1) as the layout legalizes so HPWL can settle.
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)
    # interpolate growth rate between LOWER_PCOF (low overflow) and base_mu
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * of

    # Plateau detection: if overflow has stopped improving, stop inflating
    # lambda to avoid over-penalizing density at the expense of wirelength.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if max(recent) - min(recent) < 1e-4:
            mu = min(mu, 1.0)

    # Gradient safety: if gradients blow up, damp the density push.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0)

    next_lambda = current_lambda * mu

    # Hard bound to required output range.
    if next_lambda != next_lambda:  # NaN guard
        next_lambda = 1.0
    return float(min(max(next_lambda, 0.01), 50.0))