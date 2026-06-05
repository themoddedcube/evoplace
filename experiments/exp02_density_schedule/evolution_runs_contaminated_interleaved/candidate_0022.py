def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base annealing factor: aggressive early, gentle late (cells settle).
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push lambda up while bins are congested,
    # ease off once spreading is nearly complete so HPWL can refine.
    of = overflow if overflow == overflow else 1.0  # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow over recent history.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[-min(5, len(overflow_history))]
        trend = prev - recent  # positive => overflow decreasing (good)

    # If overflow is decreasing steadily, relax the penalty growth.
    # If stalled or rising, accelerate it.
    if trend > 0.005:
        adapt = 1.0 - 0.5 * min(trend, 0.05)      # slow down growth
    elif trend < -0.001:
        adapt = 1.0 + 0.5 * min(-trend, 0.05)     # speed up growth
    else:
        adapt = 1.0

    # Late-stage fine-tuning: once overflow is low, bias toward decay
    # for accurate HPWL approximation rather than continued penalty.
    if of < 0.10:
        late = max(LOWER_PCOF, 1.0 - (0.10 - of) * 2.0)
    else:
        late = 1.0

    # Gradient-norm damping: if gradients are exploding, soften update.
    g = gradient_norm if gradient_norm == gradient_norm else 0.0
    damp = 1.0
    if g > 0.0:
        damp = 1.0 / (1.0 + 0.05 * max(0.0, g - 1.0))
        damp = max(0.8, min(1.0, damp))

    mu = base * adapt * late
    mu = 1.0 + (mu - 1.0) * damp

    # Clamp the per-step multiplier to avoid runaway updates.
    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))