def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule with hard clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.01

    # Sanitize inputs.
    of = overflow if (overflow is not None and overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)
    cur = current_lambda if (current_lambda is not None and current_lambda == current_lambda) else 1.0
    if cur <= 0.0:
        cur = 0.01

    # Base coefficient: strong early, gently decaying toward 1 over iterations
    # (matches the original 0.9999^iter envelope but floored higher for stability).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push density hard while cells are still poorly spread
    # (of -> 1), ease toward a near-unity multiplier as the layout legalizes (of -> 0).
    mu = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Trend damping from overflow history: react to whether spread is improving.
    if overflow_history is not None and len(overflow_history) >= 2:
        prev, last = overflow_history[-2], overflow_history[-1]
        if prev == prev and last == last:
            delta = last - prev
            if delta > 0.0:          # overflow rising -> diverging, push harder
                mu *= 1.02
            elif delta < -0.02:      # converging fast -> back off to avoid overshoot
                mu *= 0.99

    # Gradient guard: if gradients blow up, temper growth to avoid instability.
    if gradient_norm is not None and gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = cur * mu

    # Hard clamp to the legal range (prevents the runaway -> inf failure mode).
    if not (new_lambda == new_lambda):  # NaN guard
        new_lambda = cur
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)