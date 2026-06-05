def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule (DREAMPlace-style)."""
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # Sanitize inputs defensively so we always return a valid float.
    try:
        cur = float(current_lambda)
    except (TypeError, ValueError):
        cur = 1.0
    if not (cur == cur) or cur <= 0.0:  # NaN/inf/non-positive guard
        cur = 1.0

    try:
        ovf = float(overflow)
    except (TypeError, ValueError):
        ovf = 1.0
    if not (ovf == ovf):
        ovf = 1.0
    ovf = min(max(ovf, 0.0), 1.0)

    # Base multiplier: ramp lambda up, but cool the ramp as iterations grow.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction. When overflow is dropping fast, ease off
    # the density push so wirelength can settle; when it stalls, push harder.
    delta = 0.0
    if overflow_history:
        try:
            prev = float(overflow_history[-1])
            if prev == prev:
                delta = prev - ovf  # positive => overflow improving
        except (TypeError, ValueError, IndexError):
            delta = 0.0

    if delta > 0.02:
        # Good progress: slow the ramp toward the lower coefficient.
        mu = base * (1.0 - 0.5 * min(delta, 0.1))
    elif delta < -0.005:
        # Overflow rising again: push density harder.
        mu = base * 1.02
    else:
        mu = base

    # As placement converges (low overflow), anneal the multiplier toward 1.0
    # so lambda stops exploding and HPWL can be fine-tuned.
    if ovf < 0.10:
        mu = 1.0 + (mu - 1.0) * (ovf / 0.10)

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    result = cur * mu
    if not (result == result):  # final NaN guard
        result = cur
    return float(min(max(result, 0.01), 50.0))