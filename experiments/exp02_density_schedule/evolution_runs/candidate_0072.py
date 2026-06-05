def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) schedule for DREAMPlace.

    Grows lambda to drive overflow down, modulating the growth rate by how
    much overflow remains and how fast it is improving, then clamps to the
    legal range so the optimization never diverges.
    """
    LOWER = 0.01
    UPPER = 50.0

    # Sanitize inputs (guard against NaN/inf/None coming from the solver).
    cur = current_lambda
    if not (cur == cur) or cur in (float("inf"), float("-inf")):
        cur = 1.0
    cur = min(max(cur, LOWER), UPPER)

    ovf = overflow
    if not (ovf == ovf) or ovf < 0.0:
        ovf = 0.0
    ovf = min(ovf, 1.0)

    # Base per-step growth (DREAMPlace UPPER_PCOF style), annealed over time so
    # early iterations push hard and later iterations settle.
    base = 1.05 * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive boost: large overflow => grow faster, small => ease off.
    # Maps overflow in [0,1] to an extra multiplier in roughly [1.0, 1.10].
    adapt = 1.0 + 0.10 * (ovf ** 0.5)

    # Trend term: if overflow is stalling (not decreasing), push a bit harder;
    # if it is dropping quickly, relax to let HPWL refine.
    trend = 1.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            delta = prev - last  # positive => overflow improving
            if delta <= 1e-4:
                trend = 1.03      # stalled: increase penalty
            elif delta > 0.01:
                trend = 0.99      # improving fast: ease back

    mu = base * adapt * trend

    # Once placement is essentially legal, stop growing and gently relax so the
    # density penalty no longer distorts the final wirelength.
    if ovf < 0.08:
        mu = min(mu, 1.0)
    if ovf < 0.03:
        mu = 0.985

    new_lambda = cur * mu

    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return float(min(max(new_lambda, LOWER), UPPER))