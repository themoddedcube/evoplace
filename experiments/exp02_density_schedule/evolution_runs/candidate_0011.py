def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive multiplicative density-weight schedule.

    Grows lambda quickly while cells are still congested (high overflow),
    then tapers the growth as the layout spreads so HPWL can be fine-tuned
    without the density penalty overshooting. Falls back gracefully on
    degenerate inputs and always returns a value in [0.01, 50.0].
    """
    # --- sanitize inputs (guards against the inf/NaN that broke the parent) ---
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    it = max(0, int(iteration))
    ovf = _finite(overflow, 1.0)
    ovf = min(max(ovf, 0.0), 1.0)
    cur = _finite(current_lambda, 1.0)
    if cur <= 0.0:
        cur = 1.0
    gnorm = _finite(gradient_norm, 1.0)

    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95

    # Base DREAMPlace-style decaying multiplier (more aggressive early).
    base = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive boost: when many bins are over-dense, push the
    # density weight up faster; when nearly spread, ease off toward 1.0.
    # ovf in [0,1] -> factor in [LOWER_PCOF, UPPER_PCOF].
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ovf

    # Trend term: if overflow is rising vs. recent history, spread harder.
    trend = 1.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(h, ovf) for h in overflow_history[-3:]]
        prev = sum(recent) / len(recent)
        if ovf > prev + 1e-4:
            trend = 1.02
        elif ovf < prev - 1e-4:
            trend = 0.99

    mu = base * adapt * trend

    # Damp the step if gradients are exploding to keep optimization stable.
    if gnorm > 1e6:
        mu = min(mu, 1.0)

    # Keep the per-step multiplier in a sane band.
    mu = min(max(mu, 0.90), 1.10)

    new_lambda = cur * mu

    # Final clamp to the required output range.
    return float(min(max(new_lambda, 0.01), 50.0))