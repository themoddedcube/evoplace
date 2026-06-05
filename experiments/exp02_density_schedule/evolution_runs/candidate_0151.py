def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Multiplicative density-penalty update (augmented-Lagrangian style),
    # made overflow-adaptive and stall-aware, with safe clamping.

    # Sanitize inputs (guard against NaN/inf/None that cause inf HPWL).
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    it = max(0, int(iteration))
    ovf = min(max(_finite(overflow, 1.0), 0.0), 1.0)
    cur = _finite(current_lambda, 1.0)
    if cur <= 0.0:
        cur = 0.01

    # Base ramp: strong early growth that anneals toward ~1.0 (DREAMPlace UPPER_PCOF).
    UPPER_PCOF = 1.05
    base = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive boost: push harder while bins are congested,
    # ease off as the layout legalizes so HPWL can be fine-tuned.
    # ovf high  -> mu up to ~1.10 ; ovf low -> mu toward ~1.00.
    adapt = 1.0 + 0.10 * ovf

    # Stall detection: if overflow has stopped improving, nudge lambda up
    # to escape the plateau; if improving fast, relax to protect wirelength.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        prev = _finite(overflow_history[-3], ovf)
        delta = prev - ovf  # positive = overflow decreasing (good)
        if delta < 1e-4:        # stalled
            adapt *= 1.03
        elif delta > 1e-2:      # improving quickly
            adapt *= 0.985

    mu = base * adapt
    # Bound per-step multiplier for stability.
    mu = min(max(mu, 0.95), 1.15)

    next_lambda = cur * mu
    return float(min(max(next_lambda, 0.01), 50.0))