def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Base geometric warmup: strong early growth, settling toward a small floor.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction.
    # Push lambda harder while bins are congested, ease off as legality improves.
    of = overflow if overflow is not None else 1.0
    of = min(max(of, 0.0), 1.0)

    # Trend over recent history: are we making density progress?
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = overflow_history[-5:]
        trend = window[0] - window[-1]  # positive => overflow decreasing (good)

    # If overflow is stagnant/rising, accelerate; if dropping fast, decelerate.
    if trend <= 1e-4:
        accel = 1.0 + 0.5 * of            # stuck: ramp lambda to break clusters apart
    else:
        accel = 1.0 - min(0.4 * (trend / max(of, 1e-3)), 0.3)  # progressing: cool down

    mu = base_mu * accel

    # Gradient safeguard: if gradients explode, damp the multiplier toward neutral.
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = 1.0 + (mu - 1.0) * 0.5

    # Keep the multiplier in a sane band so a single step cannot blow up lambda.
    mu = min(max(mu, LOWER_PCOF), 1.10)

    new_lambda = current_lambda * mu

    # Near-legal: stop inflating, let wirelength dominate the final refinement.
    if of < 0.1:
        new_lambda = current_lambda * min(mu, 1.0 + of)

    return float(min(max(new_lambda, 0.01), 50.0))