def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule with clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Sanitize inputs (guard against NaN/inf/non-positive lambda).
    if not (current_lambda == current_lambda) or current_lambda in (
        float("inf"),
        float("-inf"),
    ) or current_lambda <= 0.0:
        current_lambda = 0.01
    of = overflow if overflow == overflow else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Base annealing factor: aggressive growth early, tapering as iterations accrue.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-trend modulation: if overflow is dropping the layout is improving,
    # so push the density weight harder; if it stalls/rises, grow gently to keep
    # gradients stable.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:
            delta = of - prev
            if delta < 0.0:
                base *= 1.0 + min(-delta, 0.05)
            else:
                base *= LOWER_PCOF + 0.05 * (1.0 - min(delta, 1.0))

    # Remaining-overlap scaling: lots of overflow -> grow more; near-legal -> ease off.
    mu = 1.0 + (base - 1.0) * (0.5 + 0.5 * of)

    new_lambda = current_lambda * mu
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda

    # Hard clamp to the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)