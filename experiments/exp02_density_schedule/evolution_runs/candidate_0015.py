def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Sanitize inputs (NaN/inf-safe)
    of = overflow
    if of != of or of in (float("inf"), float("-inf")):
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    gn = gradient_norm
    gn_bad = (gn != gn) or gn in (float("inf"), float("-inf"))

    # Baseline DREAMPlace-style decaying multiplier
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend over recent history
    delta = 0.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 2:
        a, b = overflow_history[-2], overflow_history[-1]
        if a == a and b == b:  # both finite
            delta = b - a

    # Overflow-adaptive ramp:
    #   high overflow + stalling/rising  -> push density weight harder
    #   overflow falling                 -> trust progress, gentle ramp
    #   near-converged (low overflow)    -> ease off so wirelength can refine
    if of > 0.1:
        if delta >= 0.0:
            mu = base_mu * 1.03      # not improving: increase pressure
        else:
            mu = base_mu             # improving: steady ramp
    else:
        # fine-tuning regime: relax toward 1.0 (and below) to sharpen HPWL
        mu = min(base_mu, 1.0 + 0.5 * of)
        if delta > 0.0:              # overflow creeping back up while converged
            mu = max(mu, 1.0)

    # Gradient safety: damp if gradients are exploding or invalid
    if gn_bad:
        mu = LOWER_PCOF
    elif gn > 0.0 and gn > 1e6:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Guard against NaN/inf blow-up (root cause of divergence)
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))