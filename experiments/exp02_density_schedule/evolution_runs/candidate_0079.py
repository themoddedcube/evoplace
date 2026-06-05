def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base multiplicative growth that anneals toward 1.0 as iterations progress,
    # mirroring the high-gamma-early -> low-gamma-late principle.
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive coefficient: push harder while bins are congested,
    # ease off once the layout starts to legalize.
    if overflow > 0.9:
        cong = 1.10
    elif overflow > 0.5:
        cong = 1.05
    elif overflow > 0.25:
        cong = 1.02
    else:
        cong = 1.00

    # Trend term: react to how overflow is moving. If overflow is stalling or
    # rising, increase the penalty faster; if it is dropping nicely, relax.
    trend = 1.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = float(overflow_history[-2])
        curr = float(overflow_history[-1])
        delta = prev - curr  # positive => improving
        if delta < 1e-4:
            trend = UPPER_PCOF       # stalled: accelerate
        elif delta > 0.02:
            trend = LOWER_PCOF       # improving fast: decelerate
        else:
            trend = 1.0

    # Gradient safeguard: if gradients are exploding, temper the growth to keep
    # optimization stable.
    grad_guard = 1.0
    if gradient_norm is not None and gradient_norm > 1e3:
        grad_guard = 0.97

    mu = base * cong * trend * grad_guard
    mu = min(max(mu, 0.95), 1.12)

    next_lambda = current_lambda * mu
    return float(min(max(next_lambda, 0.01), 50.0))