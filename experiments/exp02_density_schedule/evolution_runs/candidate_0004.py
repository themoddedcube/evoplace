def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # Base multiplicative growth that anneals as iterations progress,
    # mirroring DREAMPlace's density-weight subgradient ascent.
    decay = max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive scaling: push lambda up hard while many bins are
    # over-dense, ease off as the layout legalizes so we don't overshoot.
    of = overflow if overflow == overflow else 1.0  # guard against NaN
    of = min(max(of, 0.0), 1.0)

    # Trend from recent overflow history: if overflow is still dropping fast,
    # keep momentum; if it has stalled high, increase pressure.
    trend = 0.0
    if overflow_history is not None and len(overflow_history) >= 2:
        prev = overflow_history[-2]
        cur = overflow_history[-1]
        if prev == prev and cur == cur:
            trend = prev - cur  # positive => improving

    # Map overflow to a growth factor in roughly [LOWER_PCOF, UPPER_PCOF].
    # High overflow -> near UPPER (grow), low overflow -> ~1.0 (hold).
    mu = 1.0 + (UPPER_PCOF - 1.0) * decay * of

    # If overflow stalls at a high level, add a small extra nudge.
    if of > 0.5 and trend <= 1e-4:
        mu += 0.01 * decay

    # If overflow is low and improving, gently relax to refine wirelength.
    if of < 0.1 and trend >= 0.0:
        mu = min(mu, 1.0 + 0.005 * decay)

    # Gradient-norm safeguard: damp growth if gradients blow up to avoid
    # the divergence that produced inf HPWL.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e4:
            mu = min(mu, 1.0 + 0.5 * (UPPER_PCOF - 1.0))

    mu = min(max(mu, LOWER_PCOF), UPPER_PCOF)

    new_lambda = current_lambda * mu

    # Keep within the required, numerically safe range.
    if not (new_lambda == new_lambda):  # NaN fallback
        new_lambda = current_lambda
    return float(min(max(new_lambda, 0.01), 50.0))