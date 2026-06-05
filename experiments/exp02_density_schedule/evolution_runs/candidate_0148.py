def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.0

    # Base multiplicative growth that anneals toward 1.0 as we converge,
    # so the density penalty ramps fast early and stabilizes late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while overflow is high,
    # ease off once cells have spread out (low overflow -> fine HPWL tuning).
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Estimate how fast overflow is improving from recent history.
    delta = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            delta = prev - last  # positive => improving

    if of > 0.9:
        # Very congested: accelerate penalty growth.
        mu = base * 1.03
    elif of > 0.1:
        # Mid-phase: scale growth by remaining overflow.
        mu = base * (LOWER_PCOF + 0.5 * of)
        # If overflow stalls (little improvement), nudge harder.
        if delta < 1e-4:
            mu *= 1.02
    else:
        # Nearly legal: relax penalty to refine HPWL.
        mu = max(LOWER_PCOF, base * (0.95 + of))

    # Gradient safeguard: if gradients explode, damp the update.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        mu = min(mu, 1.01)

    mu = min(max(mu, 0.5), 1.10)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda or new_lambda <= 0.0:
        new_lambda = max(current_lambda, 0.01)

    return float(min(max(new_lambda, 0.01), 50.0))