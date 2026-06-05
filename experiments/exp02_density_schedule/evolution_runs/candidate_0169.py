def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # Sanitize inputs (guard against NaN/inf that produce inf HPWL)
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    it = max(0, int(iteration))
    ov = min(max(_finite(overflow, 1.0), 0.0), 1.0)
    cur = _finite(current_lambda, 1.0)
    if cur <= 0.0:
        cur = 0.01

    # Base multiplicative growth (DREAMPlace-style), gently annealed
    UPPER_PCOF = 1.05
    LOWER_PCOF = 0.95
    base = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive scaling: push harder while bins are congested,
    # ease off as the placement legalizes so HPWL can be fine-tuned.
    # ov high  -> multiplier near UPPER_PCOF (cluster/spread faster)
    # ov low   -> multiplier near LOWER_PCOF (relax penalty, refine HPWL)
    adapt = LOWER_PCOF + (UPPER_PCOF - LOWER_PCOF) * ov

    # Trend term: if overflow is rising, accelerate; if falling, decelerate.
    trend = 1.0
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 2:
        prev = _finite(overflow_history[-2], ov)
        last = _finite(overflow_history[-1], ov)
        delta = last - prev
        trend = 1.0 + max(-0.03, min(0.03, delta))

    # Damp updates when gradients explode (noisy regime near low gamma).
    gn = _finite(gradient_norm, 0.0)
    damp = 1.0 if gn <= 1e3 else 0.98

    mu = 0.5 * base + 0.5 * adapt
    mu *= trend * damp

    nxt = cur * mu
    if nxt != nxt or nxt in (float("inf"), float("-inf")):
        nxt = cur

    return float(min(max(nxt, 0.01), 50.0))