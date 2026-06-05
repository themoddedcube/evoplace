def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style multiplicative base that anneals toward 1.0 over time.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1] (guard NaN/inf that caused divergence).
    of = overflow if overflow == overflow and abs(overflow) != float("inf") else 1.0
    of = min(max(of, 0.0), 1.0)

    # Overflow trend: rising/stalled density => push lambda harder;
    # fast-dropping density => ease off so we don't overshoot the penalty.
    trend = 0.0
    if len(overflow_history) >= 2:
        prev, last = overflow_history[-2], overflow_history[-1]
        if prev == prev and last == last:
            trend = last - prev  # >0 means overflow getting worse

    # Adaptive multiplier: scale with remaining overflow, accelerate on bad trend.
    mu = base_mu * (1.0 + 0.5 * of) + 2.0 * max(trend, 0.0)

    # Late-stage / low-overflow: relax penalty growth for accurate HPWL fine-tuning.
    if of < 0.10:
        mu = min(mu, 1.0 + 0.5 * of)

    mu = min(max(mu, LOWER_PCOF), 1.10)

    # Gradient safeguard: damp the update if gradients blow up.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.0)

    next_lambda = current_lambda * mu
    if next_lambda != next_lambda or abs(next_lambda) == float("inf"):
        next_lambda = 1.0  # NaN/inf recovery
    return float(min(max(next_lambda, 0.01), 50.0))