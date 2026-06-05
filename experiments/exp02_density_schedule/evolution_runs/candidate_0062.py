def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Base DREAMPlace-style geometric growth of the density penalty.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001
    mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while bins are congested,
    # ease off as the layout legalizes so HPWL can settle.
    ovfl = overflow if overflow == overflow else 1.0  # guard against NaN
    ovfl = min(max(ovfl, 0.0), 1.0)

    if ovfl > 0.9:
        # Far from legal: accelerate spreading.
        mu *= 1.03
    elif ovfl < 0.1:
        # Nearly legal: slow growth, refine wirelength.
        mu = max(mu, LOWER_PCOF) * 0.97

    # Trend damping: if overflow is no longer decreasing, avoid runaway penalty.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        if recent[-1] >= recent[0] - 1e-4:
            mu = min(mu, 1.02)

    # Gradient safeguard: large gradients indicate instability, temper growth.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.01)

    next_lambda = current_lambda * mu
    if next_lambda != next_lambda:  # NaN fallback
        next_lambda = 1.0

    # Enforce required output range.
    return float(min(max(next_lambda, 0.01), 50.0))