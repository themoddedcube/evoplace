def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight update with bounded multiplier and hard clamping."""
    LOWER = 0.01
    UPPER = 50.0

    # Sanitize inputs (guard against NaN/inf that produced the inf result).
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    it = max(0, int(iteration))
    ov = _finite(overflow, 1.0)
    ov = min(max(ov, 0.0), 1.0)
    lam = _finite(current_lambda, 1.0)
    if lam <= 0.0:
        lam = LOWER

    # Base DREAMPlace-style geometric ramp, decaying as iterations grow.
    UPPER_PCOF = 1.05
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive shaping: when overflow is high, push harder to spread
    # cells; once density is nearly resolved, ease off so HPWL can settle.
    # Scale the *excess* of mu above 1.0 by the current overflow level.
    mu = 1.0 + (base_mu - 1.0) * (0.25 + 0.75 * ov)

    # Detect stalls in overflow reduction and nudge the multiplier up briefly.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(h, 1.0) for h in overflow_history[-3:]]
        if recent[-1] >= recent[0] - 1e-4 and ov > 0.10:
            mu *= 1.01

    # Late-stage relaxation: once overflow is small, relax weight to refine HPWL.
    if ov < 0.08:
        mu = min(mu, 1.0)

    # Bound the per-step multiplier to prevent runaway growth (the inf cause).
    mu = min(max(mu, 0.95), 1.10)

    new_lambda = lam * mu

    # Hard clamp to the required return range.
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = lam
    return float(min(max(new_lambda, LOWER), UPPER))