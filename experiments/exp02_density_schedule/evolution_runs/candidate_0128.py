def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight ramp with hard clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.02

    # Base DREAMPlace-style multiplicative growth, decaying with iteration.
    decay = 0.9999 ** float(iteration)
    if decay < 0.98:
        decay = 0.98
    mu = UPPER_PCOF * decay

    # Overflow-adaptive control: read the trend from history if available.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        delta = cur - prev
        if delta > 0.0:
            # Overflow rising -> spreading is losing; push density harder.
            mu *= 1.03
        elif delta < -0.02:
            # Overflow dropping fast -> ease off so the optimizer can refine HPWL.
            mu = LOWER_PCOF * decay

    # Near-convergence: low overflow means cells are placed; stop inflating lambda
    # so it does not dominate and wreck wirelength.
    if overflow < 0.10:
        mu = min(mu, 1.005)

    # Gradient safety: if gradients are exploding, throttle growth.
    if gradient_norm > 1.0e4:
        mu = min(mu, 1.0)

    # Guard against non-finite inputs.
    if not (mu == mu) or current_lambda != current_lambda:
        return 1.0

    new_lambda = current_lambda * mu

    # Hard clamp to the required range (prevents the runaway -> inf failure).
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0

    return float(new_lambda)