def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER, UPPER = 0.01, 50.0
    TARGET = 0.07  # stop pushing density once overflow is acceptable

    # --- overflow trend (negative delta == improving) ---
    if overflow_history and len(overflow_history) >= 2:
        delta = float(overflow_history[-1]) - float(overflow_history[-2])
    else:
        delta = 0.0

    # --- base multiplicative growth, faster when far above target ---
    err = overflow - TARGET
    if err > 0.0:
        # grow harder the more over-dense we are (bounded)
        mu = 1.05 + 1.5 * min(err, 0.4)
        # if overflow is stalling or rising, push even harder
        if delta >= -1e-4:
            mu *= 1.08
        else:
            # improving nicely -> ease off to protect wirelength
            mu *= 0.97
    else:
        # density satisfied: relax lambda so HPWL gradient dominates fine-tuning
        mu = 0.985

    # --- late-iteration damping so lambda settles instead of exploding ---
    damp = max(0.9999 ** float(iteration), 0.985)
    mu *= damp

    # gentle guard against gradient blow-up
    if gradient_norm > 0.0 and gradient_norm != gradient_norm:  # NaN guard
        mu = 0.99

    new_lambda = current_lambda * mu

    # hard clamp -> prevents the inf blow-up of unbounded multiplicative growth
    if new_lambda < LOWER:
        new_lambda = LOWER
    elif new_lambda > UPPER:
        new_lambda = UPPER

    return float(new_lambda)