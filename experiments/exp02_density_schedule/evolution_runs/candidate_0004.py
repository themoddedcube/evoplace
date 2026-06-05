def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with stability guards."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Base multiplicative growth (DREAMPlace-style), annealed with iteration.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push harder while bins are congested,
    # ease off as the placement spreads so we don't overshoot and diverge.
    of = overflow if overflow == overflow else 1.0          # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Estimate spreading progress from recent overflow trend.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        delta = prev - of                                    # >0 means improving
    else:
        delta = 0.0

    # Map overflow into [LOWER_PCOF, UPPER_PCOF]: high overflow -> stronger ramp.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of
    mu = min(mu, base)

    # If overflow stalls (barely improving) while still high, nudge harder.
    if of > 0.10 and abs(delta) < 1e-4:
        mu *= 1.02

    # If gradients blow up, damp the update to preserve stability.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0 + (mu - 1.0) * 0.5)

    # Near convergence (low overflow): hold lambda steady for fine-tuning.
    if of < 0.05:
        mu = 1.0 + (mu - 1.0) * 0.25

    new_lambda = current_lambda * mu

    # Hard clamp to the legal range.
    if new_lambda != new_lambda:                             # NaN -> reset low
        new_lambda = 0.01
    return float(min(max(new_lambda, 0.01), 50.0))