def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Iteration-annealed geometric base: strong density pressure early,
    # relaxing toward a near-neutral multiplier so HPWL can fine-tune late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow into [0, 1].
    of = overflow if (overflow is not None) else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Overflow trend over recent history (negative => congestion improving).
    if overflow_history and len(overflow_history) >= 2:
        window = overflow_history[-5:] if len(overflow_history) >= 5 else overflow_history
        delta = float(window[-1]) - float(window[0])
        steps = max(1, len(window) - 1)
        rate = delta / steps
    else:
        rate = 0.0

    # Push harder while bins stay congested AND overflow is stalling;
    # ease off once overflow is dropping so the placer can refine wirelength.
    stalling = 1.0 if rate > -1e-3 else 0.0
    pressure = 1.0 + 0.20 * of * stalling
    relax = 1.0 - 0.40 * max(0.0, -rate)

    # Gradient-norm guard: if gradients explode, damp the multiplier to stay stable.
    gn = gradient_norm if (gradient_norm is not None and gradient_norm > 0.0) else 0.0
    damp = 1.0 / (1.0 + 0.001 * gn) if gn > 0.0 else 1.0

    mu = base * pressure * relax * damp

    # Keep the per-step multiplier in a sane band.
    lo, hi = LOWER_PCOF, UPPER_PCOF * 1.12
    if mu < lo:
        mu = lo
    elif mu > hi:
        mu = hi

    new_lambda = current_lambda * mu

    # Clamp to the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)