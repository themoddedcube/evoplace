def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER, UPPER = 0.01, 50.0

    # --- sanitize inputs (guard against NaN/inf that produce inf HPWL) ---
    if current_lambda != current_lambda or current_lambda <= 0.0:
        current_lambda = 1.0
    if overflow != overflow:
        overflow = 1.0
    overflow = min(max(overflow, 0.0), 1.0)

    UPPER_PCOF = 1.05

    # DREAMPlace-style multiplicative growth, gently annealed over iterations
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # overflow-trend adaptation: push harder if stalled/rising, ease if improving
    mu = base_mu
    if overflow_history is not None and len(overflow_history) >= 2:
        prev, cur = overflow_history[-2], overflow_history[-1]
        if prev == prev and cur == cur:
            delta = cur - prev
            if delta > 0.0:
                mu = base_mu * 1.03
            elif delta < -0.001:
                mu = base_mu * 0.98

    # scale the *growth* by remaining overflow so the penalty stops climbing
    # once cells are spread — this is the key fix for runaway lambda -> inf
    mu = 1.0 + (mu - 1.0) * overflow

    new_lambda = current_lambda * mu

    if new_lambda < LOWER:
        new_lambda = LOWER
    elif new_lambda > UPPER:
        new_lambda = UPPER
    return float(new_lambda)