def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base geometric growth, annealed so late-stage steps are gentler.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive correction: push harder while bins are congested,
    # ease off once density is nearly resolved so HPWL can be fine-tuned.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        delta = overflow - prev
        if delta > 0.0:
            # overflow rising -> spreading stalled, increase penalty faster
            base_mu *= 1.0 + min(0.5, 10.0 * delta)
        else:
            # overflow falling -> progress is good, relax growth
            base_mu *= max(LOWER_PCOF, 1.0 + 2.0 * delta)

    # Congestion-weighted scaling: strong penalty early, soft near convergence.
    if overflow < 0.1:
        base_mu *= 0.97
    elif overflow > 0.6:
        base_mu = max(base_mu, UPPER_PCOF)

    # Guard against exploding/vanishing gradients destabilizing the step.
    if gradient_norm > 0.0 and gradient_norm != gradient_norm:  # NaN guard
        base_mu = 1.0
    base_mu = min(max(base_mu, 0.5), 1.5)

    new_lambda = current_lambda * base_mu

    # Sanitize: handle NaN/inf and clamp to the legal range.
    if not (new_lambda == new_lambda) or new_lambda in (float("inf"), float("-inf")):
        new_lambda = current_lambda
    return min(max(new_lambda, 0.01), 50.0)