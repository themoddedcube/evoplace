def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base RePlAce-style geometric ramp: aggressive early, gentle late.
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend from history: are we still spreading or stalling?
    if overflow_history:
        prev = overflow_history[-1]
        # average a short tail for a less noisy slope estimate
        tail = overflow_history[-3:]
        avg_prev = sum(tail) / len(tail)
        delta = overflow - avg_prev          # <0 means overflow improving
    else:
        prev = overflow
        delta = 0.0

    # Adaptive multiplier: push lambda harder when overflow is high and
    # not improving; ease off as it stalls or starts rising (overshoot).
    # delta normalized by a reference step (~0.02 overflow change).
    ref = 0.02
    coef = UPPER_PCOF - (UPPER_PCOF - LOWER_PCOF) * max(0.0, min(1.0, delta / ref + 0.5))

    # Scale the push by remaining overflow: little overflow -> small change.
    overflow_factor = max(0.1, min(1.0, overflow / 0.10))
    mu = 1.0 + (coef * decay - 1.0) * overflow_factor

    # Gradient-norm guard: if gradients explode, damp the increase to stay stable.
    if gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.02)

    new_lambda = current_lambda * mu

    # Hard clamp to the required range.
    return float(max(0.01, min(50.0, new_lambda)))