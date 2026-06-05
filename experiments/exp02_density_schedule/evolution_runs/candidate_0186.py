def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # Base multiplicative growth that anneals as iterations progress so the
    # density penalty ramps hard early (cluster cells) then eases off late.
    decay = max(0.9999 ** float(iteration), 0.98)
    mu = UPPER_PCOF * decay

    # Overflow-adaptive modulation (DREAMPlace-style): if overflow is shrinking
    # we are converging, so slow the lambda ramp; if it stalls or grows, push
    # harder to keep spreading cells.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        delta = overflow - prev  # negative => improving
        # map delta in roughly [-0.05, +0.05] to a scale in [LOWER, UPPER]
        scale = 1.0 + max(-0.5, min(0.5, delta * 10.0)) * 0.05
        mu *= scale

    # When overflow is already low, density is nearly satisfied: stop inflating
    # lambda so the optimizer can fine-tune wirelength instead of overspreading.
    if overflow < 0.10:
        mu = min(mu, LOWER_PCOF + 0.01)
    elif overflow < 0.20:
        mu = min(mu, 1.03)

    # Guard against exploding gradients destabilizing the ramp.
    if gradient_norm is not None and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.01)

    new_lambda = current_lambda * mu

    # Hard clamp to the required output range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)