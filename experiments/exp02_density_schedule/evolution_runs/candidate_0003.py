def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style decaying growth cap: aggressive early, gentler later.
    base = max(0.9999 ** float(iteration), 0.98)

    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = max(0.0, min(1.0, of))

    # Overflow trend: are we still spreading, or starting to diverge?
    if overflow_history and len(overflow_history) >= 2:
        recent = float(overflow_history[-1]) - float(overflow_history[-2])
    else:
        recent = 0.0

    target = 0.10
    if of > target:
        # Cells still overlapping: grow lambda, harder the further from target,
        # so density penalty clusters/legalizes before HPWL fine-tuning.
        mu = UPPER_PCOF * base * (1.0 + 0.5 * (of - target))
    else:
        # Near legal: ease off (mu <= 1) to let HPWL gradients refine placement.
        mu = max(LOWER_PCOF, 1.0 - 0.5 * (target - of))

    # Divergence guard: if overflow is climbing, stop inflating lambda.
    if recent > 0.02:
        mu = min(mu, 1.0)

    # Exploding gradients => damp growth to avoid runaway lambda (the inf case).
    if gradient_norm == gradient_norm and gradient_norm > 0.0 and current_lambda > 0.0:
        if gradient_norm * current_lambda > 1e3:
            mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:  # NaN -> reset low
        new_lambda = 0.01
    return max(0.01, min(50.0, new_lambda))