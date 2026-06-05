def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # DREAMPlace-style multiplicative density-weight update, made overflow-adaptive.
    # Base envelope: aggressive early growth that anneals toward neutral, so the
    # penalty ramps while cells are still clustered and eases as layout settles.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend from history (negative delta == spreading is converging).
    if len(overflow_history) >= 2:
        delta = overflow - overflow_history[-2]
    elif overflow_history:
        delta = overflow - overflow_history[-1]
    else:
        delta = 0.0

    # Smoothed recent slope to avoid reacting to single-iteration noise.
    if len(overflow_history) >= 4:
        slope = (overflow_history[-1] - overflow_history[-4]) / 3.0
    else:
        slope = delta

    mu = UPPER_PCOF * base

    # Stalling / rising overflow: density not being enforced fast enough -> push.
    if slope > -0.0005 and overflow > 0.10:
        mu *= 1.05
    # Healthy convergence: spreading well, let HPWL settle without over-penalizing.
    elif delta < -0.001:
        mu *= 0.98

    # Near-final regime: low overflow means geometry is essentially placed, so
    # hold the penalty nearly flat and let low-gamma HPWL refinement dominate.
    if overflow < 0.08:
        mu = min(mu, 1.02)
    elif overflow < 0.15:
        mu = min(mu, 1.04)

    # Very noisy gradients -> damp the step to keep the trajectory stable.
    if gradient_norm > 0.0 and gradient_norm > 5.0:
        mu = 1.0 + (mu - 1.0) * 0.7

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))