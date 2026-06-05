def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    ovf = float(overflow)
    if ovf != ovf:                      # NaN guard
        ovf = 1.0
    ovf = min(max(ovf, 0.0), 1.0)

    # Base annealed multiplier: aggressive early, gentle late (DREAMPlace style)
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction from recent trend:
    # overflow stalling -> push lambda harder; dropping fast -> ease off.
    mu = base_mu
    if overflow_history and len(overflow_history) >= 2:
        prev = float(overflow_history[-1])
        if prev == prev:                # not NaN
            ref = max(prev, 1e-3)
            rate = (prev - ovf) / ref   # >0 means overflow improving
            mu = base_mu * (1.0 - 0.5 * rate)

    # Damp growth as placement converges (low overflow) to avoid over-penalizing.
    if ovf < 0.10:
        mu = 1.0 + (mu - 1.0) * (ovf / 0.10)

    # Stabilize against exploding/vanishing gradients.
    if gradient_norm != gradient_norm or gradient_norm == float("inf"):
        mu = 1.0

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:        # NaN guard on result
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))