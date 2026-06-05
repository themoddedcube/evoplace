def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth of the density penalty (DREAMPlace-style),
    # with a slowly relaxing floor so growth never fully stalls.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push the penalty up hard while many
    # bins are still over-dense, ease off as the layout legalizes so the
    # solver can fine-tune wirelength without density over-shooting.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    overflow_gain = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)

    # Trend term: if overflow is rising or sticking, accelerate; if it is
    # dropping steadily, relax the multiplier toward neutral.
    trend = 1.0
    if overflow_history and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[-1] - recent[0]
        if delta > 1e-4:
            trend = 1.03          # overflow worsening -> stronger penalty
        elif delta < -1e-3:
            trend = 0.99          # legalizing well -> back off slightly

    # Gradient safeguard: if gradients are exploding, temper growth to keep
    # the optimization stable.
    grad_safe = 1.0
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            grad_safe = 0.97

    mu = base_mu * overflow_gain * trend * grad_safe

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:   # NaN guard
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))