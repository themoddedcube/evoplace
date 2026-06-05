def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric ramp (DREAMPlace-style), gently annealed with iteration.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push harder while bins are congested,
    # ease off as the layout legalizes so we don't over-spread late.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from recent history: is overflow still dropping?
    trend = 0.0
    if overflow_history:
        n = min(3, len(overflow_history))
        recent = overflow_history[-n:]
        if len(recent) >= 2 and recent[0] == recent[0]:
            trend = recent[0] - recent[-1]  # >0 means improving

    # When overflow is high, accelerate; when it stalls (small trend
    # but high overflow) accelerate more; when low/improving, relax.
    accel = 1.0 + 0.25 * of
    if of > 0.4 and trend < 0.01:
        accel += 0.15  # break out of a stalled, congested state
    if of < 0.10:
        accel = min(accel, 1.0 - 0.5 * (0.10 - of))  # decay toward refinement

    mu = base * accel
    mu = min(max(mu, LOWER_PCOF), 1.30)

    next_lambda = current_lambda * mu
    return float(min(max(next_lambda, 0.01), 50.0))