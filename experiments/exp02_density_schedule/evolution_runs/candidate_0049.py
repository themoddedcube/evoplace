def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base RePlAce-style decaying multiplier (high early, ~1 late)
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth: push density harder while many bins are
    # over-filled, ease off as the layout legalizes.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow over recent history: if stalling, accelerate.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:
            trend = of - prev                                # >0 means worsening

    # Map overflow to a multiplier in [LOWER_PCOF, UPPER_PCOF].
    # High overflow -> stronger lambda increase; low overflow -> gentle.
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # If overflow refuses to drop (stall/increase), nudge growth up a touch.
    if trend >= 0.0 and of > 0.1:
        pcof = min(pcof * 1.01, 1.08)

    # Gradient-norm safeguard: if gradients explode, damp the increase
    # to keep optimization stable.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 0.0 and gn > 1e6:
        pcof = min(pcof, 1.0)

    mu = pcof * base_mu

    new_lambda = current_lambda * mu

    # Clamp to the required range.
    if new_lambda != new_lambda:                             # NaN guard
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0

    return float(new_lambda)