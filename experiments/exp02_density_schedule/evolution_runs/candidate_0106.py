def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.0001

    # Base geometric decay of the multiplier (DREAMPlace-style warm-up -> settle)
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push lambda harder while bins are congested,
    # ease off (toward 1.0) as the layout legalizes so HPWL can be fine-tuned.
    of = overflow if overflow == overflow else 1.0   # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Map overflow in [0,1] to a multiplier weight in [LOWER_PCOF, UPPER_PCOF].
    # High overflow -> full UPPER_PCOF; low overflow -> near 1.0 (stop inflating).
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (of ** 0.5)

    # Trend awareness: if overflow is rising vs recent history, accelerate;
    # if steadily falling, decelerate to avoid overshoot.
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-2]
        if recent > prev:
            mu *= 1.01            # congestion worsening -> firmer push
        elif recent < prev * 0.97:
            mu *= 0.995           # legalizing well -> back off slightly

    # Gradient safeguard: damp the update if gradients are exploding,
    # so the penalty does not destabilize the descent.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 0.0:
        damp = 1.0 / (1.0 + 0.001 * gn)
        mu = 1.0 + (mu - 1.0) * max(damp, 0.5)

    new_lambda = current_lambda * mu

    # Clamp to the legal range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)