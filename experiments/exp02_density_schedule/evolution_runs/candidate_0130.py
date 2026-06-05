def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that produced divergence).
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    ov = _finite(overflow, 1.0)
    ov = min(max(ov, 0.0), 1.0)
    lam = _finite(current_lambda, LO)
    if lam <= 0.0:
        lam = LO
    gnorm = _finite(gradient_norm, 0.0)

    # ePlace-style multiplicative growth, but modulated by overflow so the
    # density penalty ramps hard while cells overlap and coasts once spread.
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base decaying ceiling on the per-step multiplier (as in the original).
    base = max(0.9999 ** float(iteration), 0.98)

    # Overflow trend: are we still improving, plateauing, or backsliding?
    trend = 0.0
    if overflow_history:
        recent = [
            _finite(h, ov)
            for h in overflow_history[-5:]
        ]
        if len(recent) >= 2:
            trend = recent[0] - recent[-1]  # >0 means overflow dropping (good)

    # Map overflow to growth strength: high overflow -> push lambda up fast,
    # low overflow -> nearly hold lambda to avoid over-penalizing & diverging.
    push = ov ** 0.5  # in [0,1]; emphasizes mid-range overflow

    if trend < -1e-4:
        # Overflow getting worse: ease off so density gradient doesn't explode.
        mu = LOWER_PCOF + 0.05 * push
    else:
        # Overflow flat or improving: grow, scaled by how much spreading remains.
        mu = 1.0 + (UPPER_PCOF - 1.0) * base * push

    # Gradient-norm safety valve: if gradients are blowing up, damp the penalty.
    if gnorm > 0.0 and gnorm > 1e6:
        mu = min(mu, 1.0)

    new_lambda = lam * mu

    # Final clamp + NaN/inf guard.
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = lam
    return float(min(max(new_lambda, LO), HI))