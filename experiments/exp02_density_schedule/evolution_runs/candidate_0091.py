def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Time-decaying base: aggressive early, gentle late.
    base = max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1].
    of = overflow if overflow == overflow else 1.0
    of = min(max(of, 0.0), 1.0)

    # Overflow trend from history (positive = improving).
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev, cur = overflow_history[-2], overflow_history[-1]
        if prev == prev and cur == cur:
            trend = prev - cur

    # Scale multiplier by overflow severity: high overflow -> push density harder.
    mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * base * (0.5 + 0.5 * of)

    # Stalled overflow -> nudge the penalty up.
    if trend < 1e-3:
        mu *= 1.01

    # Damp the ramp if gradients are exploding (stability guard).
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0 + 0.5 * (UPPER_PCOF - 1.0))

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = current_lambda

    # Hard clamp prevents the divergence that produced inf.
    return float(min(max(new_lambda, 0.01), 50.0))