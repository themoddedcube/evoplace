def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule with hard clamp."""
    LOWER, UPPER = 0.01, 50.0

    # Base DREAMPlace-style multiplier: aggressive early, gentle late.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive blend: high overflow -> push lambda up faster so cells
    # spread; low overflow -> ease off and let HPWL/gradients fine-tune.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)
    blend = UPPER_PCOF * of + LOWER_PCOF * (1.0 - of)
    mu = blend * decay

    # Trend awareness: if overflow stalled or rebounded, give an extra nudge;
    # if it is dropping nicely, relax to avoid overshooting the density penalty.
    if len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[-1] - recent[0]
        if delta > -1e-4:          # not improving
            mu *= 1.02
        elif delta < -0.05:        # improving quickly
            mu *= 0.99

    # Damp updates when gradients explode to keep the optimization stable.
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    next_lambda = current_lambda * mu
    if next_lambda != next_lambda:  # NaN fallback
        next_lambda = current_lambda

    return float(min(max(next_lambda, LOWER), UPPER))