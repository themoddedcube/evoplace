def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.0

    # Base multiplicative growth that cools as iterations progress
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push lambda up hard while bins are congested,
    # ease off as the layout legalizes so we don't over-penalize density.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)
    overflow_gain = 1.0 + 0.7 * of  # 1.0 (legal) .. 1.7 (fully congested)

    # Plateau detection: if overflow has stopped improving, accelerate growth
    # to escape the stall; if it is dropping fast, decelerate for fine-tuning.
    plateau_gain = 1.0
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-4:
            plateau_gain = 1.15          # stalled -> stronger density pressure
        elif improvement > 5e-3:
            plateau_gain = 0.92          # improving fast -> slow down

    # Gradient-norm damping: when gradients explode, temper lambda growth
    # to keep the optimization stable; when calm, allow normal progression.
    g = gradient_norm if gradient_norm == gradient_norm else 0.0
    if g > 0.0:
        grad_damp = 1.0 / (1.0 + 0.02 * max(g - 1.0, 0.0))
    else:
        grad_damp = 1.0

    mu = base_mu * overflow_gain * plateau_gain * grad_damp
    mu = min(max(mu, LOWER_PCOF * 0.5), 1.8)  # bound per-step change

    new_lambda = current_lambda * mu

    # Clamp to the legal range
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)