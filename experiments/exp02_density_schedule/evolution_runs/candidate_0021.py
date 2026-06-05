def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight ramp with safety clamps."""
    # Sanitize inputs
    of = overflow if (overflow == overflow and overflow >= 0.0) else 1.0
    cl = current_lambda if (current_lambda == current_lambda and current_lambda > 0.0) else 1.0

    # Base DREAMPlace-style multiplicative growth, decaying with iteration.
    base = 1.05 * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive boost: push harder while many bins are over-filled,
    # ease off (toward gentle growth) as the layout legalizes.
    # of ~ 1.0 -> faster ramp; of ~ 0.1 -> near-neutral.
    boost = 1.0 + 0.30 * max(min(of, 1.0), 0.0)

    # Detect stagnation in overflow: if overflow stopped improving, nudge harder
    # to escape plateaus; if it is dropping fast, relax to avoid overshoot.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 4:
        recent = overflow_history[-4:]
        delta = recent[0] - recent[-1]  # positive => improving
        if delta < 1e-4:
            boost *= 1.05          # stalled: increase pressure
        elif delta > 0.05:
            boost *= 0.97          # fast drop: ease back

    # Gradient-norm guard: if gradients explode, temper the growth.
    gn = gradient_norm if (gradient_norm == gradient_norm and gradient_norm >= 0.0) else 0.0
    if gn > 1e3:
        boost *= 0.95

    mu = base * boost

    # Keep the per-step multiplier in a sane band to prevent blow-up / collapse.
    mu = max(0.95, min(mu, 1.10))

    new_lambda = cl * mu

    # Final hard clamp to the required output range.
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = cl
    return float(max(0.01, min(new_lambda, 50.0)))