def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """ ... """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth, gently decaying as the run matures.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push density harder while many bins are
    # over-full, ease off as overflow approaches the target so wirelength
    # can settle without being crushed by the density penalty.
    target_overflow = 0.10
    if overflow > target_overflow:
        of_factor = 1.0 + min((overflow - target_overflow) * 2.0, 0.5)
    else:
        of_factor = max(1.0 - (target_overflow - overflow) * 2.0, LOWER_PCOF)

    # Plateau breaker: if overflow has stalled over the recent window, nudge
    # the penalty up to escape the stall; if it is dropping fast, relax.
    if len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        delta = recent[0] - recent[-1]
        if delta < 1e-4:
            of_factor *= 1.05
        elif delta > 0.05:
            of_factor *= 0.97

    # Gradient guard: if gradients are exploding, slow the lambda ramp.
    if gradient_norm > 0.0 and gradient_norm > 1e3:
        of_factor *= 0.9

    mu = base_mu * of_factor
    new_lambda = current_lambda * mu

    # Numerical safety: reject NaN/Inf, then clamp into the legal range.
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))