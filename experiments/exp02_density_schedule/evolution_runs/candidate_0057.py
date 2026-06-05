def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty multiplier with hard clamping."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric growth that anneals as iterations accumulate.
    decay = max(0.9999 ** float(iteration), 0.98)
    mu = UPPER_PCOF * decay

    # Adapt to the overflow trajectory (DREAMPlace-style subgradient feel):
    # if overflow is dropping, ease the penalty so wirelength can refine;
    # if it stalls or rises, push the multiplier harder.
    if overflow_history:
        prev = overflow_history[-1]
        delta = prev - overflow  # >0 means overflow is improving
        if delta > 0.0:
            ease = min(max(delta / max(prev, 1e-6), 0.0), 1.0)
            mu = LOWER_PCOF + (mu - LOWER_PCOF) * (1.0 - 0.5 * ease)
        else:
            mu = mu * 1.02

    # Late-stage fine-tuning: once nearly legal, stop inflating lambda.
    if overflow < 0.10:
        mu = min(mu, 1.005)

    # Stability guard against gradient blow-up (the failure mode that
    # sends lambda -> inf and HPWL -> inf).
    if gradient_norm > 1e3:
        mu = min(mu, 1.0)
    mu = min(max(mu, 0.5), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))