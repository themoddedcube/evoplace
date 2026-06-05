def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    """Overflow-adaptive density-weight schedule with bounded growth."""
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that produce inf HPWL).
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    ovf = max(0.0, min(1.0, _finite(overflow, 1.0)))
    lam = _finite(current_lambda, 1.0)
    gnorm = _finite(gradient_norm, 0.0)

    # Seed a sane starting weight if we were handed garbage.
    if lam <= 0.0:
        lam = 1.0

    # Base multiplicative growth, capped well below runaway exponential blow-up.
    # Grow harder while bins are still overfilled, ease off as the layout legalizes.
    base = 1.02 + 0.10 * ovf            # in [1.02, 1.12]

    # Plateau detection: if overflow has stopped improving, push a little harder
    # to escape the stall; if it is dropping fast, relax to avoid overshoot.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(h, ovf) for h in overflow_history[-3:]]
        delta = recent[0] - recent[-1]   # positive => overflow decreasing (good)
        if delta < 1e-4:                 # stalled
            base *= 1.03
        elif delta > 0.02:               # improving quickly
            base *= 0.98

    # Gradient safety: if gradients are exploding, slow the weight increase.
    if gnorm > 1e3:
        base = min(base, 1.01)

    # Late-stage fine-tuning: once nearly legal, freeze growth so HPWL can settle.
    if ovf < 0.08:
        base = min(base, 1.005)

    new_lambda = lam * base

    # Hard clamp to the required output range.
    if new_lambda != new_lambda:         # NaN guard
        new_lambda = lam
    return float(max(LO, min(HI, new_lambda)))