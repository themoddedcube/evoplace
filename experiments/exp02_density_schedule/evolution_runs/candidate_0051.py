def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative decay (DREAMPlace-style), slightly faster late.
    base_mu = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive coefficient: when the layout is still congested
    # (high overflow) keep gamma smooth (mu near UPPER); once overflow
    # drops, accelerate toward accurate-HPWL regime (mu toward LOWER).
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    pcof = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Detect overflow stagnation: if recent overflow is barely changing,
    # push harder toward low gamma to escape the plateau and fine-tune.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        delta = abs(recent[-1] - recent[0])
        if delta < 1e-3:
            pcof *= 0.97

    # Gradient safety: if gradients explode, soften the step toward 1.0
    # to avoid destabilizing the optimizer.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    if gn > 1e3:
        pcof = 1.0 + (pcof - 1.0) * 0.5

    mu = pcof * base_mu

    new_lambda = current_lambda * mu
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)