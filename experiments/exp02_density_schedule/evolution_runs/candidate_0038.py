def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Sanitize inputs (guard against NaN/inf leaking into the multiplier).
    of = overflow
    if of != of or of > 1e30 or of < -1e30:
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    cur = current_lambda
    if cur != cur or cur > 1e30 or cur < -1e30 or cur <= 0.0:
        cur = 1.0

    # Base multiplicative growth that decays toward ~1 as placement proceeds:
    # ramp the density penalty hard early, then stabilize for fine-tuning.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive pressure: many over-dense bins -> push lambda up
    # faster; near-legal placement -> ease off so HPWL can settle.
    mu = base * (1.0 + 0.5 * (of - 0.10))

    # Overflow trend: if it is stalling or rising, apply stronger pressure.
    if len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            delta = last - prev
            if delta > 1e-4:
                mu *= 1.04
            elif delta < -5e-3:
                mu *= 0.98  # legalizing quickly -> relax growth

    # Gradient safeguard: damp growth if gradients explode (numerical safety).
    gn = gradient_norm
    if gn == gn and gn > 1e3:
        mu = min(mu, 1.0)

    # Keep the per-step multiplier in a sane band.
    mu = min(max(mu, LOWER_PCOF), 1.10)

    new_lambda = cur * mu

    # Robust final clamp to the required output range; handle NaN/inf.
    if new_lambda != new_lambda or new_lambda > 1e30 or new_lambda < -1e30:
        new_lambda = cur
    return min(max(new_lambda, 0.01), 50.0)