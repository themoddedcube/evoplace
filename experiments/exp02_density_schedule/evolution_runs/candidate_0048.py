def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs
    it = float(iteration) if iteration is not None else 0.0
    of = overflow if (overflow is not None and overflow == overflow) else 1.0
    of = min(max(float(of), 0.0), 1.0)
    cur = current_lambda if (current_lambda is not None and current_lambda == current_lambda) else 1.0
    cur = min(max(float(cur), 0.01), 50.0)

    # Base annealed growth (RePlAce-style multiplicative ramp that decays over time)
    base_mu = UPPER_PCOF * max(0.9999 ** it, 0.98)

    # Overflow-adaptive: grow lambda harder while the layout is still
    # overlapped, ease off as bins legalize toward the density target.
    of_factor = 1.0 + 0.06 * of

    # Overflow trend: accelerate on stall/regression, decelerate when
    # overflow is falling quickly so we don't over-penalize density.
    trend = 0.0
    if overflow_history:
        prev = overflow_history[-1]
        if prev is not None and prev == prev:
            trend = of - float(prev)
    if trend > 0.0:
        of_factor *= 1.015
    elif trend < -0.02:
        of_factor *= 0.99

    # Gradient guard: if gradients are exploding, throttle lambda growth.
    gn = gradient_norm if (gradient_norm is not None and gradient_norm == gradient_norm) else 0.0
    if gn > 1e3:
        of_factor *= 0.98

    mu = base_mu * of_factor
    mu = min(max(mu, LOWER_PCOF), 1.10)

    new_lambda = cur * mu

    # Final safety: reject NaN/inf, clamp to legal range.
    if not (new_lambda == new_lambda) or new_lambda in (float('inf'), float('-inf')):
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))