def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # NaN/inf guards on inputs
    cl = current_lambda
    if cl != cl or cl in (float("inf"), float("-inf")):
        cl = 1.0
    of = overflow
    if of != of:
        of = 1.0
    of = min(max(of, 0.0), 1.0)

    # DREAMPlace-style decaying growth envelope: strong push early, gentle late
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push lambda harder while bins are congested,
    # relax toward 1.0 as the placement spreads (low overflow -> fine-tuning).
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Overflow trend: if congestion is already falling, avoid overshoot.
    if overflow_history and len(overflow_history) >= 2:
        h1 = overflow_history[-1]
        h2 = overflow_history[-2]
        if h1 == h1 and h2 == h2:
            trend = h1 - h2
            if trend < 0.0:
                pcof = min(pcof, 1.0 + 0.5 * (UPPER_PCOF - 1.0))
            elif trend > 0.02:
                pcof = min(UPPER_PCOF, pcof + 0.5 * (UPPER_PCOF - 1.0))

    # Gradient safeguard: damp growth when gradients explode (stability).
    gn = gradient_norm
    if gn == gn and gn > 1.0:
        damp = 1.0 / (1.0 + (gn - 1.0) * 0.01)
        pcof = 1.0 + (pcof - 1.0) * damp

    mu = pcof * base
    new_lambda = cl * mu

    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cl

    return float(min(max(new_lambda, 0.01), 50.0))