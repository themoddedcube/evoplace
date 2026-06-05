def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with momentum and safety clamps."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric ramp (the proven DREAMPlace backbone), gently annealed.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Estimate overflow trend from recent history.
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-min(5, len(overflow_history)):]
        delta = recent[0] - recent[-1]          # >0 means overflow improving
        span = max(len(recent) - 1, 1)
        rate = delta / span                      # avg per-iter improvement
    else:
        rate = 0.0

    # Adaptive factor: push harder when overflow is high and stalling,
    # ease off once density is being resolved quickly.
    if overflow > 0.10:
        if rate <= 1e-4:
            # Stalled with high overflow -> accelerate the penalty.
            adapt = 1.10
        elif rate > 0.01:
            # Improving fast -> let the base ramp do the work, avoid overshoot.
            adapt = 1.00
        else:
            adapt = 1.05
    else:
        # Low overflow: enter fine-tuning, relax the penalty growth.
        if rate < 0:
            # Overflow creeping back up -> nudge penalty up to hold legality.
            adapt = 1.03
        else:
            adapt = LOWER_PCOF

    # Damp updates when gradients are exploding to keep optimization stable.
    if gradient_norm > 0.0:
        grad_damp = 1.0 / (1.0 + max(0.0, gradient_norm - 1.0) * 0.05)
    else:
        grad_damp = 1.0

    mu = 1.0 + (base_mu * adapt * grad_damp - 1.0)

    # Keep the multiplier in a sane band so a single step can't blow up/collapse.
    if mu < 0.90:
        mu = 0.90
    elif mu > 1.15:
        mu = 1.15

    new_lambda = current_lambda * mu

    # Enforce the hard return bounds.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return new_lambda