def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base ePlace-style geometric growth, gently annealed over iterations.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive growth rate: push hard while cells are still
    # spread out (high overflow), ease off as the layout legalizes so we
    # don't over-penalize density once cells are nearly placed.
    of = overflow if overflow == overflow else 1.0      # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow: if it is stalling/rising, accelerate the penalty;
    # if it is dropping fast, slow down to let HPWL gradients dominate.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        trend = recent[0] - recent[-1]                  # >0 means improving

    if of > 0.20:
        # Far from convergence: density must keep tightening.
        accel = 1.0 + 0.5 * (of - 0.20)
        if trend <= 1e-4:                               # stalled -> push more
            accel *= 1.10
        mu = base * accel
    else:
        # Near convergence: anneal the penalty so wirelength gradients win,
        # scaling growth down smoothly with remaining overflow.
        mu = 1.0 + (base - 1.0) * (of / 0.20)
        if trend > 1e-3:                                # improving nicely
            mu = max(mu, LOWER_PCOF)

    # Gradient-norm safety: if gradients explode, damp the multiplier.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * (1e3 / gradient_norm))

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.5)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))