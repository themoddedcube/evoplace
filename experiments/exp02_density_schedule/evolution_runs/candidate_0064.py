def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base DREAMPlace-style multiplicative step, gently slowing with iteration.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: when cells are still congested (high overflow)
    # keep lambda climbing to push spreading; once overflow relaxes, slow the
    # growth so HPWL can be fine-tuned without over-penalizing density.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from recent overflow history: if overflow is stalling/rising, be more
    # aggressive; if it is dropping fast, ease off.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-1]
        prev = overflow_history[min(len(overflow_history) - 1, 4)]
        # prev is an earlier sample (clamped index); positive => overflow falling
        if prev == prev and recent == recent:
            trend = float(prev) - float(recent)

    # density_factor in roughly [LOWER_PCOF, UPPER_PCOF]: high overflow -> push up.
    density_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of

    # Slow growth when overflow is dropping quickly (good progress).
    if trend > 0.02:
        density_factor *= 0.985

    mu = base * density_factor

    # Gradient-norm safety: if gradients explode, damp the step to stay stable.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu *= 0.97

    # Keep the per-step multiplier in a sane band.
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = current_lambda * mu
    return float(min(max(new_lambda, 0.01), 50.0))