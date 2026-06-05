def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with bounded growth."""
    LOWER = 0.01
    UPPER = 50.0

    # Base multiplicative growth (DREAMPlace-style), annealed over iterations.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push density force harder while bins are
    # congested, ease off as the layout legalizes so HPWL can refine.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow: if it is no longer dropping, lean on a stronger push;
    # if it is dropping fast, relax toward the lower coefficient.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        delta = of - overflow_history[-1] if False else (prev - of)
    else:
        delta = 0.0

    # Map overflow in [0,1] to a multiplier between LOWER_PCOF and UPPER_PCOF.
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    if delta <= 0.0:                                         # stalled / rising
        adapt = max(adapt, 0.5 * (adapt + UPPER_PCOF))
    mu = 0.5 * (base + adapt)

    # Gradient safeguard: if gradients blow up, do not amplify further.
    if gradient_norm == gradient_norm and gradient_norm > 1e6:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Hard clamp to the required range to avoid divergence (inf HPWL).
    if new_lambda != new_lambda:                            # NaN -> reset
        new_lambda = 1.0
    return float(min(max(new_lambda, LOWER), UPPER))