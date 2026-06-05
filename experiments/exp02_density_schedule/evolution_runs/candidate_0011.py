def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base annealing multiplier: strong early growth, gentle decay so the
    # density weight keeps rising while cells are still clustered.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term. High overflow => cells still overlapping =>
    # push lambda up faster. Low overflow => near-legal => ease off so the
    # wirelength term can fine-tune without density over-spreading.
    of = max(0.0, min(1.0, overflow))
    # smooth map: ~UPPER_PCOF at of=1, ~LOWER_PCOF at of=0
    overflow_mu = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * (of ** 0.5)

    # Trend term: if overflow stalls or rises over recent history, the
    # current weight is too weak -> accelerate; if it drops quickly, relax.
    trend = 1.0
    if overflow_history is not None and len(overflow_history) >= 3:
        recent = overflow_history[-3:]
        delta = recent[-1] - recent[0]  # negative = improving
        if delta > -0.005:
            # plateau or worsening: boost growth
            trend = 1.0 + min(0.10, abs(delta) * 2.0 + 0.02)
        else:
            # improving well: temper growth slightly
            trend = 1.0 - min(0.05, (-delta))

    # Gradient safeguard: damp multiplicative growth when gradients explode
    # to keep the optimization stable.
    grad_damp = 1.0
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            grad_damp = 0.97
        elif gradient_norm > 1e2:
            grad_damp = 0.99

    mu = base * overflow_mu * trend * grad_damp

    # Late-stage convergence: once nearly legal, stop inflating lambda.
    if of < 0.08:
        mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)