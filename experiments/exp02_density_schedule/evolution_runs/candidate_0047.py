def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # DREAMPlace-style multiplicative growth, decaying with iteration
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push density penalty harder while cells are spread,
    # ease off as the layout legalizes so HPWL can be fine-tuned.
    if overflow > 0.10:
        of_factor = 1.0 + min(0.5 * (overflow - 0.10), 0.10)
    else:
        of_factor = max(0.90, 1.0 - 0.5 * (0.10 - overflow))

    # Stagnation / divergence detection from overflow trend.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[0] - recent[-1]
        if 0.0 <= delta < 1e-3:        # stuck: add pressure
            trend_factor = 1.04
        elif delta < 0.0:              # overflow rising: too aggressive
            trend_factor = 0.97
        else:                          # improving steadily
            trend_factor = 1.0
    else:
        trend_factor = 1.0

    # Gradient safeguard: soften update if gradients explode or are invalid.
    if gradient_norm == gradient_norm and gradient_norm < 1e3:
        grad_factor = 1.0
    else:
        grad_factor = 0.95

    mu = base_mu * of_factor * trend_factor * grad_factor
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF * 1.1)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))