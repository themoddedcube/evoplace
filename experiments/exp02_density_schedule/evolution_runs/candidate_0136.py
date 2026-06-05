def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that produce inf HPWL).
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    cl = _finite(current_lambda, 1.0)
    ov = _finite(overflow, 1.0)
    gn = _finite(gradient_norm, 1.0)
    cl = min(max(cl, LO), HI)
    ov = min(max(ov, 0.0), 1.0)

    # Base multiplicative growth (DREAMPlace-style), capped so it never explodes.
    UPPER_PCOF, LOWER_PCOF = 1.05, 1.0
    base = LOWER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive: push harder while many bins are over-dense,
    # relax growth as the layout legalizes so fine-tuning is stable.
    ov_gain = (UPPER_PCOF - LOWER_PCOF) * min(ov / 0.10, 1.0)
    mu = base + ov_gain

    # Trend awareness: if overflow stalls or worsens, grow a touch more;
    # if it is dropping fast, ease off to avoid overshoot.
    if len(overflow_history) >= 2:
        prev = _finite(overflow_history[-2], ov)
        delta = ov - prev
        if delta > 1e-4:
            mu *= 1.01
        elif delta < -1e-2:
            mu *= 0.99

    # Gradient safeguard: damp growth when gradients are large/noisy.
    if gn > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * 0.5)

    mu = min(max(mu, 0.98), 1.10)
    new_lambda = cl * mu

    if new_lambda != new_lambda:  # NaN guard
        new_lambda = cl
    return min(max(new_lambda, LO), HI)