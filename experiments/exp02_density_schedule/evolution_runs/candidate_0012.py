def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Geometric base growth that gently decays as placement matures.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push the density penalty harder when
    # overflow stalls, ease off when it is already dropping fast.
    mu = base_mu
    if overflow_history and len(overflow_history) >= 2:
        delta = float(overflow_history[-1]) - float(overflow_history[-2])
        # delta < 0  -> overflow falling (good)  -> smaller multiplier
        # delta > 0  -> stuck / regressing       -> larger multiplier
        adapt = 1.0 + max(-0.05, min(0.05, delta * 4.0))
        mu = base_mu * adapt

    # Late-stage cooldown: once cells are well spread, stop inflating
    # the penalty so HPWL can be fine-tuned without distortion.
    if overflow < 0.10:
        mu = min(mu, LOWER_PCOF + 0.05 * (overflow / 0.10))

    # Gradient safety: damp growth if gradients blow up or are non-finite.
    if not (gradient_norm == gradient_norm) or gradient_norm == float("inf"):
        mu = 1.0
    elif gradient_norm > 1e4:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Hard clamp to the contractually valid range (prevents inf blow-up).
    if not (new_lambda == new_lambda):
        new_lambda = current_lambda
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0

    return float(new_lambda)