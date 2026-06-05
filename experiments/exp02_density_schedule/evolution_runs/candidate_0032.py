def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight (lambda) growth with safety clamps."""
    # Sanitize inputs.
    of = overflow if (overflow is not None) else 1.0
    of = min(max(of, 0.0), 1.0)
    cur = current_lambda if (current_lambda is not None and current_lambda > 0.0) else 0.01
    grad = gradient_norm if (gradient_norm is not None and gradient_norm > 0.0) else 1.0

    # Base multiplicative growth that decays with iteration: push density hard
    # early (cells still spreading), gently anneal late for HPWL fine-tuning.
    base = 1.045 * max(0.9999 ** float(iteration), 0.98)

    # Overflow term: many over-dense bins -> larger step; near-legal -> ~1.0.
    of_boost = 1.0 + 0.12 * of

    # Trend term: if overflow stalls or rises, push harder; if it is falling
    # fast the layout is legalizing, so ease the penalty growth.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        window = min(len(overflow_history), 5)
        trend = overflow_history[-1] - overflow_history[-window]
    trend = min(max(trend, -0.5), 0.5)
    trend_boost = 1.0 + 0.20 * trend

    # Gradient safety: damp lambda growth when gradients are exploding to
    # avoid the divergence that produced inf HPWL.
    if grad > 1e4:
        grad_damp = 0.5
    elif grad > 1e3:
        grad_damp = 0.8
    else:
        grad_damp = 1.0

    mu = base * of_boost * trend_boost * grad_damp

    # Keep per-step multiplier conservative and monotone-ish.
    mu = min(max(mu, 0.95), 1.12)

    new_lambda = cur * mu
    return float(min(max(new_lambda, 0.01), 50.0))