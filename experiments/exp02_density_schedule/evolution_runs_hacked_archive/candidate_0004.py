def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ ... """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric warm-up (DREAMPlace-style), gently annealed with iteration
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive acceleration: push density harder while bins are congested,
    # ease off as the layout legalizes so wirelength can be fine-tuned.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow over recent history: if overflow is stalling/rising,
    # increase lambda faster; if it is dropping fast, relax.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-min(5, len(overflow_history)):]
        trend = recent[-1] - recent[0]  # >0 means overflow worsening

    # Map overflow + trend into a multiplier in [LOWER_PCOF, UPPER_PCOF + margin]
    congestion = of + max(trend, 0.0)
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF + 0.10) * min(congestion, 1.0)

    mu = 0.5 * base_mu + 0.5 * adapt

    # Gradient safeguard: if gradients explode, damp lambda growth to avoid divergence.
    gn = gradient_norm if gradient_norm == gradient_norm else 1.0
    if gn > 0.0 and gn != float("inf"):
        if gn > 5.0:
            mu = min(mu, 1.0)
        elif gn > 2.0:
            mu = min(mu, 1.02)

    # Late-stage convergence: once nearly legal, hold lambda steady for clean fine-tuning.
    if of < 0.10:
        mu = min(mu, 1.0 + 0.5 * of)

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF + 0.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))