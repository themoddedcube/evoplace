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

    # RePlAce-style annealed multiplicative base (high early, eases late)
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # overflow trend: negative delta == density improving
    if len(overflow_history) >= 2:
        delta = float(overflow) - float(overflow_history[-2])
    else:
        delta = 0.0

    if overflow <= TARGET_OVERFLOW:
        # density satisfied: freeze lambda so low-gamma fine-tuning minimizes HPWL
        mu = 1.0
    elif delta >= 0.0:
        # overflow stalled or rising: push harder to spread cells
        mu = base_mu
    else:
        # overflow already dropping: temper growth to avoid overshoot
        mu = max(LOWER_PCOF, 1.0 + (base_mu - 1.0) * 0.5)

    # divergence guard: NaN/inf or exploding gradients -> do not inflate
    if not (gradient_norm == gradient_norm) or gradient_norm > 1e12:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if not (new_lambda == new_lambda):  # NaN fallback
        new_lambda = current_lambda

    return float(min(50.0, max(0.01, new_lambda)))