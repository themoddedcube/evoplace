def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Annealed aggressiveness: large steps early, gentler late (fine-tuning phase).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push the density penalty harder while bins are congested,
    # ease off as cells spread out so HPWL can settle.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    coef = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Trend-aware: if overflow has stalled, increase pressure; if it is dropping
    # quickly, relax so the placement is not over-penalized into a worse minimum.
    if len(overflow_history) >= 2:
        delta = overflow_history[-2] - overflow             # >0 means improving
        if delta <= 1e-4:
            coef *= 1.03                                     # stalled -> push
        elif delta > 5e-3:
            coef *= 0.98                                     # fast progress -> ease

    # Gradient safeguard: damp growth when gradients blow up to avoid instability.
    if gradient_norm > 0.0 and gradient_norm == gradient_norm:
        if gradient_norm > 1e4:
            coef *= 0.97

    mu = coef * base
    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))