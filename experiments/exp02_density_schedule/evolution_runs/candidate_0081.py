def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base annealed growth rate: aggressive early, gentle late.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when density overflow is still high we
    # push the penalty up faster; as overflow drops we ease off so the
    # optimizer can fine-tune wirelength without over-spreading.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Estimate overflow trend from history (positive => improving).
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            trend = prev - last  # >0 means overflow decreasing

    # Density-driven factor in roughly [LOWER_PCOF, UPPER_PCOF].
    density_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # If overflow has stalled (little/no improvement) but is still high,
    # nudge growth up to break the plateau.
    if of > 0.1 and trend < 1e-4:
        density_factor *= 1.02
    # If overflow is improving nicely, relax to protect HPWL accuracy.
    elif trend > 1e-3:
        density_factor *= 0.99

    # Gradient-norm safeguard: damp growth if gradients are exploding.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        density_factor = min(density_factor, 1.0)

    mu = 0.5 * base_mu + 0.5 * (base_mu * density_factor)

    # Late-stage convergence: once overflow is essentially resolved, stop
    # growing the penalty so the placer can settle on accurate wirelength.
    if of < 0.06:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to the legal range.
    if new_lambda != new_lambda:
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))