def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-penalty multiplier with clamped, NaN-safe output."""
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00

    # DREAMPlace-style geometric ramp, gently annealed with iteration so the
    # penalty grows fast early (cells still spreading) and tapers later.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Sanitize and bound overflow to [0, 1].
    of = overflow if overflow == overflow else 1.0  # NaN guard
    of = min(max(of, 0.0), 1.0)

    # Overflow-adaptive scaling: push the penalty harder while bins are
    # congested, ease off as the placement legalizes to avoid over-penalizing
    # an already-spread layout (the regime where low overflow needs accuracy).
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (0.3 + 0.7 * of)

    # Trend from history: if overflow stalls or worsens, add a little pressure.
    if overflow_history and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        if prev == prev and cur == cur and cur >= prev:
            mu *= 1.0 + min(cur - prev, 0.05)

    # Damp growth when gradients explode (noisy late-stage signal).
    if gradient_norm == gradient_norm and gradient_norm > 1e3:
        mu = min(mu, 1.01)

    next_lambda = current_lambda * mu

    # NaN guard then hard clamp to the required output range.
    if not (next_lambda == next_lambda):
        next_lambda = current_lambda
    return float(min(max(next_lambda, 0.01), 50.0))