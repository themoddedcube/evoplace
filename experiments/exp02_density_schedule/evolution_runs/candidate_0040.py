def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Multiplicative density-weight update (DREAMPlace-style mu),
    # made overflow-adaptive and hard-clamped to avoid the runaway
    # blow-up that drove HPWL to inf.
    LOWER = 0.01
    UPPER = 50.0

    # Seed a sane starting value if the optimizer handed us garbage.
    base = current_lambda
    if not (base == base) or base <= 0.0:   # NaN / non-positive guard
        base = 1.0

    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow: are we still spreading, or has it stalled?
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        if prev == prev:
            trend = prev - of   # positive => overflow shrinking (good)

    # Push density harder while bins are congested, ease off as the
    # placement legalizes. mu in [~1.0, ~1.06].
    mu = 1.0 + 0.06 * of

    # If overflow has plateaued (barely moving) but is still high,
    # nudge a bit more weight to break the stall.
    if of > 0.10 and abs(trend) < 1e-3:
        mu += 0.01

    # Slow the climb once we are nearly legal so HPWL can settle.
    if of < 0.10:
        mu = 1.0 + 0.5 * (mu - 1.0)

    # Damp growth if gradients are exploding (noisy/unstable step).
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = 1.0 + 0.5 * (mu - 1.0)

    new_lambda = base * mu
    return min(max(new_lambda, LOWER), UPPER)