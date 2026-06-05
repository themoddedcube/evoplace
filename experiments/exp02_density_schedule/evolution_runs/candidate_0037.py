def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    TARGET_OVERFLOW = 0.10

    # Annealed multiplicative base (slows growth as iterations accumulate)
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize overflow
    of = overflow if overflow == overflow else 1.0
    if of < 0.0:
        of = 0.0
    elif of > 1.0:
        of = 1.0

    # Progress signal from history (positive => overflow improving)
    progress = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        if prev == prev:
            progress = prev - of

    if of > TARGET_OVERFLOW:
        # Density still too high: scale growth by distance above target
        excess = (of - TARGET_OVERFLOW) / (1.0 - TARGET_OVERFLOW)
        mu = base_mu * (1.0 + 0.5 * excess)
        if progress <= 0.0:
            mu *= 1.10  # stalled spreading: push density penalty harder
    else:
        # Density target met: stop inflating lambda, refine HPWL
        mu = min(base_mu, 1.0)

    # Guard against gradient blow-up
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0)

    # Clamp the per-step multiplier
    if mu < LOWER_PCOF:
        mu = LOWER_PCOF
    elif mu > 1.10:
        mu = 1.10

    new_lambda = current_lambda * mu
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda

    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)