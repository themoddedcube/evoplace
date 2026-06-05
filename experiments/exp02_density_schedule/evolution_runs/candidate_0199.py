def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs (guard against NaN/inf that produce inf HPWL).
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    cl = current_lambda
    if cl != cl or cl in (float("inf"), float("-inf")):
        cl = 0.01
    cl = min(max(cl, 0.01), 50.0)

    # Base multiplicative growth that decays toward 1 as iterations proceed:
    # ramp density pressure quickly early, then settle to avoid runaway blow-up.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: positive => overflow is dropping (placement legalizing).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        k = min(len(overflow_history), 5)
        prev = overflow_history[-k]
        recent = overflow_history[-1]
        if prev == prev and recent == recent:
            trend = prev - recent

    # Adapt growth: high overflow or stalled progress -> push harder;
    # near-legal & improving -> ease toward mu~1 to refine wirelength.
    stalled = 1.0 if trend <= 1e-4 else 0.0
    relax = 1.0 if of < 0.10 else 0.0
    adapt = 1.0 + 0.6 * of + 0.05 * stalled - 0.4 * relax
    adapt = min(max(adapt, 0.0), 1.5)

    mu = 1.0 + (base_mu - 1.0) * adapt
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.2)

    new_lambda = cl * mu
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return float(min(max(new_lambda, 0.01), 50.0))