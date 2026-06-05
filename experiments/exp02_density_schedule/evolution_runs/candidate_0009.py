def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplier: anneal the growth rate so lambda ramps up
    # aggressively early (push cells apart) and gently late (fine-tune HPWL).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction: compare recent overflow trend.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        # Rate of overflow reduction (positive = improving spread).
        delta = prev - cur
        if prev > 1e-12:
            rel = delta / prev
        else:
            rel = 0.0

        if delta <= 0.0:
            # Overflow stalled or grew: push density penalty harder.
            mu = base * (1.0 + min(0.10, 0.5 * abs(rel) + 0.02))
        else:
            # Overflow shrinking: scale growth down as we converge,
            # easing off near low overflow so gradients stay accurate.
            ease = max(LOWER_PCOF, 1.0 - 0.8 * rel)
            mu = base * ease
    else:
        mu = base

    # When overflow is already low, slow lambda growth to refine wirelength.
    if overflow < 0.10:
        mu = 1.0 + (mu - 1.0) * (overflow / 0.10)

    new_lambda = current_lambda * mu

    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)