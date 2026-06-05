def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base ePlace-style decaying upper bound on the growth factor.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push lambda up hard while cells still
    # overlap heavily, then ease off so the late stage can minimize HPWL
    # under accurate (low-gamma) gradients instead of over-penalizing density.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    if of > 0.9:
        # Very congested: grow aggressively to disperse cells.
        mu = base * 1.08
    elif of > 0.1:
        # Smoothly interpolate the growth factor with overflow.
        frac = (of - 0.1) / 0.8
        mu = base * (0.97 + 0.11 * frac)
    else:
        # Near-legal: stop inflating density weight, let wirelength dominate.
        mu = LOWER_PCOF + 0.05 * (of / 0.1)

    # Plateau detection: if overflow has stalled, nudge lambda up to break out.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        spread = max(recent) - min(recent)
        if spread < 0.01 and of > 0.1:
            mu *= 1.03

    # Gradient safeguard: damp updates when gradients explode to stay stable.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))