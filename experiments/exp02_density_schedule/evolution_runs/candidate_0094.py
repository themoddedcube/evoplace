def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LOWER, UPPER = 0.01, 50.0

    # Sanitize inputs (guard against nan/inf that produced the inf result).
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    it = max(0, int(iteration))
    ovf = min(1.0, max(0.0, _finite(overflow, 1.0)))
    gnorm = max(0.0, _finite(gradient_norm, 0.0))
    cur = _finite(current_lambda, LOWER)
    cur = min(UPPER, max(LOWER, cur))

    # Base multiplicative growth, decaying with iteration so early iters ramp
    # the density weight fast and late iters fine-tune (DREAMPlace-style).
    base = 1.0 + 0.05 * max(0.9999 ** float(it), 0.85)

    # Overflow-adaptive boost: push harder while many bins remain over-dense,
    # ease off as the layout legalizes so HPWL can settle.
    overflow_gain = 0.6 * ovf

    # Trend term: if overflow is stalling (not decreasing), accelerate.
    trend = 0.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(v, ovf) for v in overflow_history[-3:]]
        delta = recent[0] - recent[-1]          # positive => overflow falling
        if delta <= 1e-4:
            trend = 0.15                         # stalled: give it a nudge
        elif delta > 1e-2:
            trend = -0.05                        # converging well: relax

    mu = base + overflow_gain + trend

    # Damp growth when gradients explode to keep optimization stable.
    if gnorm > 1e3:
        mu = min(mu, 1.02)

    mu = min(1.8, max(1.0, mu))

    nxt = cur * mu

    # Hard clamp to the required range.
    if nxt != nxt or nxt in (float("inf"), float("-inf")):
        nxt = UPPER
    return float(min(UPPER, max(LOWER, nxt)))