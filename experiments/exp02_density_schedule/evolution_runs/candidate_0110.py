def lambda_schedule(
    iteration: int,
    overflow: float,
    overflow_history: list,
    gradient_norm: float,
    current_lambda: float,
) -> float:
    # --- sanitize inputs (guard against NaN/inf/None -> output inf) ---
    def _finite(x, default):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return default
        if xf != xf or xf in (float("inf"), float("-inf")):
            return default
        return xf

    it = max(0, int(iteration) if iteration is not None else 0)
    ovf = _finite(overflow, 1.0)
    ovf = min(max(ovf, 0.0), 1.0)
    lam = _finite(current_lambda, 1.0)
    if lam <= 0.0:
        lam = 0.01
    gnorm = _finite(gradient_norm, 1.0)

    # --- overflow trend: are we still making density progress? ---
    trend = 0.0
    if overflow_history:
        hist = [_finite(h, ovf) for h in overflow_history[-5:]]
        if len(hist) >= 2:
            trend = hist[0] - hist[-1]  # positive => overflow decreasing (good)

    # --- base DREAMPlace-style growth (sub-linear decay of step) ---
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # --- overflow-adaptive scaling ---
    # high overflow  -> push lambda up faster (cells still spread out)
    # low overflow   -> ease off so HPWL/legalization can settle
    if ovf > 0.20:
        mu = base_mu * (1.0 + 0.6 * (ovf - 0.20))
    else:
        # near-legal: gentle growth, taper toward 1.0 as overflow vanishes
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (ovf / 0.20)

    # if overflow stagnates (no downward trend) while still high, nudge harder
    if ovf > 0.10 and trend <= 1e-4:
        mu *= 1.10

    # damp growth when gradients explode (numerical safety)
    if gnorm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    # clamp multiplier to a sane band
    mu = min(max(mu, 0.95), 1.20)

    new_lambda = lam * mu

    # --- final clamp to required range ---
    if new_lambda != new_lambda:  # NaN guard
        new_lambda = lam
    return float(min(max(new_lambda, 0.01), 50.0))