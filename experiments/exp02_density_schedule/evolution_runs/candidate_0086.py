def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ Overflow-adaptive multiplicative density-penalty schedule. """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Base ramp: aggressive early, decaying toward a floor (DREAMPlace-style).
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while bins are congested,
    # ease off as the layout legalizes so we don't over-penalize and diverge.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    if of > 0.10:
        # Still congested: scale growth with how far above target we are.
        mu = base_mu * (1.0 + 0.5 * (of - 0.10))
    else:
        # Nearly legal: relax penalty growth toward neutral for fine HPWL tuning.
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (of / 0.10)

    # Stagnation check: if overflow stopped improving, nudge penalty up a touch.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4 and of > 0.10:
            mu *= 1.02

    # Gradient safeguard: if gradients explode, damp growth to keep stability.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.01)

    mu = min(max(mu, 0.98), 1.10)
    next_lambda = current_lambda * mu

    return float(min(max(next_lambda, 0.01), 50.0))