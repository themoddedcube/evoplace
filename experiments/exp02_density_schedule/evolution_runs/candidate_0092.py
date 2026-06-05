def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style geometric multiplier, decaying toward a 0.98 floor.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow (guard NaN/inf, clamp to [0, 1]).
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # Recent overflow trend: positive => still legalizing.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = overflow_history[-5:]
        if window[0] == window[0] and window[-1] == window[-1]:
            trend = window[0] - window[-1]

    # Adaptive penalty growth.
    if of > 0.10:
        # High overflow: push harder, and accelerate further on a stall.
        accel = 1.0 + 0.5 * of
        if trend < 0.01:
            accel *= 1.10
    else:
        # Near-legal: ease off so HPWL can settle (anneal multiplier down).
        accel = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of / 0.10)

    mu = base_mu * accel

    # Damp when gradients explode to avoid divergence.
    gn = gradient_norm
    if gn == gn and gn > 1.0e3:
        mu = min(mu, 1.02)

    new_lambda = current_lambda * mu

    # NaN guard and legal-range clamp.
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))