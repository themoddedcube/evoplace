def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Baseline DREAMPlace-style subgradient step: aggressive early, decaying.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow trend from history (positive delta => density is improving).
    delta = 0.0
    if overflow_history:
        n = len(overflow_history)
        if n >= 4:
            # Short-window slope: average of recent improvements, robust to noise.
            recent = overflow_history[-4:]
            delta = (recent[0] - recent[-1]) / 3.0
        elif n >= 2:
            delta = overflow_history[-2] - overflow

    # Gradient-norm guard: when gradients explode, damp lambda growth to stay stable.
    grad_damp = 1.0
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 5.0:
            grad_damp = 0.97
        elif gradient_norm < 0.5:
            grad_damp = 1.02  # gradients quiet -> safe to push density harder

    if overflow > 0.10:
        # Far from legal: keep spreading cells. If progress stalls, push harder.
        if delta < 0.0005:
            mu = base_mu * 1.04   # stalled / worsening -> escalate density penalty
        else:
            mu = base_mu          # healthy descent -> nominal escalation
    elif overflow > 0.05:
        # Transition band: ramp down growth so HPWL approximation can sharpen.
        mu = base_mu * 0.98
    else:
        # Near-legal: ease the penalty so cells settle and fine HPWL dominates.
        mu = max(LOWER_PCOF, base_mu * 0.95)

    mu *= grad_damp

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))