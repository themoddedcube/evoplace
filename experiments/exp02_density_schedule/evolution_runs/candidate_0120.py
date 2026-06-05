def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Base geometric growth (DREAMPlace-style), decaying with iteration
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push harder when bins are still congested,
    # ease off as the layout legalizes so we don't over-penalize density.
    ov = overflow if overflow == overflow else 1.0      # NaN guard
    ov = min(max(ov, 0.0), 1.0)

    # Map overflow in [0,1] to a multiplier in [LOWER_PCOF, UPPER_PCOF].
    adaptive = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ov

    # Detect stagnation/oscillation in recent overflow to avoid runaway growth.
    if isinstance(overflow_history, list) and len(overflow_history) >= 3:
        recent = [h for h in overflow_history[-3:] if h == h]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            if trend > -1e-4:           # overflow not improving -> gentler push
                adaptive = min(adaptive, base_mu)

    mu = 0.5 * (base_mu + adaptive)

    # Dampen if gradients are exploding to keep optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * 0.5)

    next_lambda = current_lambda * mu
    if not (next_lambda == next_lambda):                # NaN -> reset low
        next_lambda = 0.01

    return float(min(max(next_lambda, 0.01), 50.0))