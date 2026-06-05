def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # --- bounds for the returned multiplier target ---
    LAMBDA_MIN = 0.01
    LAMBDA_MAX = 50.0

    # Sanitize inputs (NaN/inf guards) so we never propagate inf -> inf.
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
    cur = min(max(cur, LAMBDA_MIN), LAMBDA_MAX)
    gnorm = _finite(gradient_norm, 1.0)

    # Base geometric growth (DREAMPlace-style density-force ramp), but
    # capped so lambda cannot run away to inf. This is the key fix over the
    # unbounded current_lambda * mu form.
    UPPER_PCOF = 1.05
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive scaling: while cells are still spread out (high
    # overflow) we want the density force to climb faster; once the layout
    # has compacted (low overflow) we ease off so wirelength can settle.
    # overflow ~1.0 -> faster ramp; overflow ~0.0 -> gentle ramp.
    ovf_gain = 0.85 + 0.30 * ovf  # in [0.85, 1.15]

    # Plateau detection: if overflow has stopped improving, push lambda a
    # touch harder to break the stall.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 4:
        recent = [_finite(h, ovf) for h in overflow_history[-4:]]
        improvement = recent[0] - recent[-1]
        if improvement < 1e-3:
            ovf_gain *= 1.05  # nudge through the plateau

    # Gradient safeguard: if gradients explode, damp the multiplier to keep
    # the optimization stable (prevents the inf blow-up).
    grad_damp = 1.0
    if gnorm > 1e3:
        grad_damp = 0.95

    mu = 1.0 + (base_mu - 1.0) * ovf_gain * grad_damp

    new_lambda = cur * mu

    # Hard clamp to the legal range and final NaN/inf guard.
    if new_lambda != new_lambda or new_lambda in (float("inf"), float("-inf")):
        new_lambda = cur
    return float(min(max(new_lambda, LAMBDA_MIN), LAMBDA_MAX))