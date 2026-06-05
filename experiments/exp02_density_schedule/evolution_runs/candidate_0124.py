def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.005

    # Sanitize inputs so a bad value can never propagate to inf/nan.
    if current_lambda != current_lambda or current_lambda <= 0.0:
        current_lambda = 0.01
    of = overflow if (overflow == overflow) else 1.0
    of = min(max(of, 0.0), 1.0)

    # Base DREAMPlace-style decaying multiplier (mu in (0.98, 1.05]).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push density weight harder while many bins are
    # over-full, and ease off (mu -> ~1) as the layout legalizes so HPWL can
    # settle without overshoot.
    growth = LOWER_PCOF + (base - LOWER_PCOF) * of

    # Detect stalled overflow (plateau) and give a mild extra push to escape it.
    if len(overflow_history) >= 5:
        recent = overflow_history[-5:]
        spread = max(recent) - min(recent)
        if spread < 1e-3 and of > 0.1:
            growth *= 1.02

    # Damp growth when gradients explode to keep the optimization stable.
    gn = gradient_norm if (gradient_norm == gradient_norm) else 0.0
    if gn > 1e4:
        growth = 1.0 + (growth - 1.0) * 0.5

    new_lambda = current_lambda * growth

    # Hard clamp to the required output range.
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))