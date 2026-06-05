def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05

    # Base growth: density weight rises over iterations, fastest early on,
    # mirroring the proven ePlace/DREAMPlace ramp.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)
    mu = base_mu

    # Overflow-adaptive correction. Use the recent trend so the penalty
    # pushes harder when overflow stalls and eases when it drops fast.
    if isinstance(overflow_history, list) and len(overflow_history) >= 2:
        prev = float(overflow_history[-2])
        ref = max(abs(prev), 1e-3)
        rel = (prev - overflow) / ref           # >0 => overflow decreasing (good)
        adj = 1.0 - 2.0 * rel                    # decreasing fast -> smaller mu
        adj = min(max(adj, 0.75), 1.30)
        mu = base_mu * adj

    # Near convergence: relax the penalty so low-gamma fine-tuning can
    # minimize HPWL without re-spreading cells.
    if overflow < 0.10:
        mu = min(mu, 1.02)
    if overflow < 0.05:
        mu = min(mu, 1.005)

    # Gradient safety: never amplify the penalty on exploding/NaN gradients.
    if gradient_norm != gradient_norm:           # NaN
        mu = 1.0
    elif gradient_norm > 1e4:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu
    if new_lambda != new_lambda:                 # NaN guard
        new_lambda = current_lambda

    return float(min(max(new_lambda, 0.01), 50.0))