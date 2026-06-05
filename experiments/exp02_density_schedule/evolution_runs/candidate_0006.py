def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive geometric growth of the density multiplier.

    Mirrors DREAMPlace's subgradient lambda update but modulates the
    growth factor mu by how fast overflow is actually falling:
      - overflow dropping fast  -> grow slowly (let the gradient work)
      - overflow stalled high    -> grow faster (push cells apart harder)
    A late-iteration / low-overflow taper freezes lambda so the final
    HPWL fine-tuning isn't disturbed.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Sanitize inputs (guard against NaN/inf that produced the failing run).
    if not (current_lambda == current_lambda) or current_lambda <= 0.0:
        current_lambda = 1.0
    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Baseline decay of the cap, as in the original schedule.
    base = max(0.9999 ** float(iteration), 0.98)

    # Estimate recent overflow trend from history (positive = improving).
    delta = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-1]
        prev2 = overflow_history[-min(5, len(overflow_history))]
        if prev == prev and prev2 == prev2:
            delta = prev2 - prev  # how much overflow fell over the window

    # Map trend -> growth factor in [LOWER_PCOF, UPPER_PCOF].
    # Big improvement -> small mu; stall -> large mu.
    progress = max(min(delta * 20.0, 1.0), 0.0)
    mu = UPPER_PCOF - (UPPER_PCOF - LOWER_PCOF) * progress
    mu *= base

    # Overflow gate: once the layout is nearly legal, stop inflating lambda
    # so the optimizer can minimize HPWL without density blowing it up.
    if of < 0.10:
        mu = min(mu, 1.0)
    elif of < 0.20:
        mu = min(mu, 1.01)

    new_lambda = current_lambda * mu

    # Hard clamp to the required output range.
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))