def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule (DREAMPlace-style)."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base annealing factor: aggressive early, gentle late.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Estimate how fast overflow is dropping using recent history.
    delta = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-min(len(overflow_history), 4)]
        delta = recent - prev  # negative => overflow improving

    # Reference scale for normalizing the overflow trend.
    ref = max(overflow, 0.05)

    # If overflow stalls (delta ~ 0 or positive) push lambda harder so the
    # density penalty escapes the plateau; if it is dropping nicely, ease off
    # to keep gradients clean for HPWL fine-tuning.
    trend = delta / ref
    adapt = 1.0 - trend  # stall -> >1 (more push), improving -> <1 (ease)

    # Gradient-norm guard: very large gradients => slow the ramp to stay stable.
    if gradient_norm is not None and gradient_norm > 0.0:
        g = 1.0 / (1.0 + 0.10 * gradient_norm)
        adapt = adapt * (0.5 + 0.5 * g)

    mu = base * adapt
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    # Late-stage relaxation: once overflow is essentially resolved, stop
    # inflating lambda and let HPWL dominate.
    if overflow < 0.10:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))