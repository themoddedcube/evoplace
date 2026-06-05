def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base annealed growth (DREAMPlace-style multiplicative ramp).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when many bins are over-dense we want the
    # density penalty to climb faster to spread cells; once the layout is
    # nearly legal (low overflow) we ease off so HPWL can fine-tune.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)
    overflow_gain = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)

    # Stagnation detection: if overflow has barely moved over recent history
    # the optimizer is stuck — nudge the multiplier up to break the plateau.
    stagnation = 1.0
    if overflow_history and len(overflow_history) >= 4:
        window = overflow_history[-4:]
        spread = max(window) - min(window)
        if spread < 1e-3 and of > 0.1:
            stagnation = 1.03

    # Gradient guard: very large gradients mean a noisy step, so temper growth
    # to avoid overshooting the density penalty.
    grad_damp = 1.0
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            grad_damp = 0.985

    mu = base * overflow_gain * stagnation * grad_damp

    # Keep growth bounded per step for stability.
    mu = min(max(mu, 0.95), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))