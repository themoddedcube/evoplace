def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty multiplier.

    Ramp lambda up aggressively while cells are still spread out (high
    overflow), then anneal the growth rate as the layout legalizes so the
    wirelength term can dominate during fine-tuning.
    """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base ePlace-style decaying ceiling on the growth factor.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive term: push harder when overflow is high, ease off
    # (and eventually shrink) once the placement is nearly legal.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)
    of_factor = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * of  # in [0.95, 1.05]

    # Trend of overflow: if it is stalling, accelerate slightly to break out.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        last = overflow_history[-1]
        if prev == prev and last == last:
            trend = prev - last  # positive => overflow improving
    stall_boost = 1.0 if trend > 1e-4 else 1.02

    # Gradient safeguard: if gradients explode, damp the multiplier.
    gn = gradient_norm if gradient_norm == gradient_norm else 0.0
    grad_damp = 1.0 if gn < 1e6 else 0.99

    mu = 0.5 * base_mu + 0.5 * of_factor
    mu *= stall_boost * grad_damp

    # Keep the per-step multiplier sane so lambda neither freezes nor blows up.
    mu = min(max(mu, 0.92), 1.10)

    new_lambda = current_lambda * mu

    # Final hard clamp to the required output range.
    if new_lambda != new_lambda:        # NaN guard
        new_lambda = 1.0
    return float(min(max(new_lambda, 0.01), 50.0))