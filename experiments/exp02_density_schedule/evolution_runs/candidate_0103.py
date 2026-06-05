def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    LO, HI = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that produced inf HPWL).
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
    ov = min(1.0, max(0.0, ov))
    cur = _finite(current_lambda, LO)
    cur = min(HI, max(LO, cur))
    gnorm = _finite(gradient_norm, 1.0)

    # Base DREAMPlace-style multiplicative growth: faster early, gentler late.
    UPPER_PCOF = 1.05
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # Overflow-adaptive boost: when many bins are still over-dense, push
    # density weight up harder; once spreading is nearly done, slow growth so
    # the optimizer can fine-tune wirelength instead of over-densifying.
    # ov ~ 1.0 -> boost toward ~1.06; ov ~ 0.1 -> mu ~1.0 (near steady).
    overflow_gain = 0.10
    mu = 1.0 + (base_mu - 1.0) * (0.15 + 0.85 * ov) + overflow_gain * (ov - 0.5) * (ov - 0.5 > 0)

    # Trend awareness: if overflow has stopped decreasing, density weight is
    # too weak — nudge growth up to break the stall.
    if isinstance(overflow_history, (list, tuple)) and len(overflow_history) >= 3:
        recent = [_finite(h, ov) for h in overflow_history[-3:]]
        if recent[-1] >= recent[0] - 1e-4:
            mu *= 1.01

    # Gradient safety: if gradients explode, damp the update to stay stable.
    if gnorm > 1e3:
        mu = min(mu, 1.0 + (mu - 1.0) * 0.5)

    nxt = cur * mu

    # Hard clamp to the legal range — this is what was missing before
    # (unbounded growth -> >50.0 -> inf HPWL).
    if nxt != nxt or nxt in (float("inf"), float("-inf")):
        nxt = cur
    return min(HI, max(LO, nxt))