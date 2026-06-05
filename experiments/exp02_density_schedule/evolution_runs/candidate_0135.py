def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base DREAMPlace-style geometric growth, decaying with iteration.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push lambda harder while density is bad,
    # ease off as the layout legalizes so HPWL gradients dominate late.
    of = overflow if overflow is not None else 1.0
    of = min(max(of, 0.0), 1.0)
    overflow_gain = 1.0 + 0.30 * of  # up to +30% growth at full overflow

    # Trend term: if overflow is stalling (not decreasing), accelerate;
    # if it is dropping fast, relax to protect wirelength.
    trend = 1.0
    if overflow_history is not None and len(overflow_history) >= 4:
        recent = sum(overflow_history[-2:]) / 2.0
        prev = sum(overflow_history[-4:-2]) / 2.0
        delta = prev - recent  # positive => improving
        if delta < 1e-4:
            trend = 1.08          # stalled: push density penalty
        elif delta > 1e-2:
            trend = 0.97          # improving fast: ease off

    # Gradient-aware damping: very large gradients mean the step is already
    # aggressive, so avoid compounding instability.
    gn = gradient_norm if gradient_norm is not None else 0.0
    grad_damp = 1.0 / (1.0 + 0.05 * max(gn - 5.0, 0.0))

    mu = base_mu * overflow_gain * trend * grad_damp
    mu = min(max(mu, LOWER_PCOF), 1.20)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))