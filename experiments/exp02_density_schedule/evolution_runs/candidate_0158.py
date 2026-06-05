def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    LOWER, UPPER = 0.01, 50.0

    # Sanitize inputs (guard against NaN/inf that produced the failing run).
    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x or x == float("inf") or x == float("-inf"):
            return default
        return x

    lam = _finite(current_lambda, 1.0)
    ovf = _finite(overflow, 1.0)
    gnorm = _finite(gradient_norm, 1.0)

    # Seed a sane starting weight if the optimizer handed us garbage.
    if lam <= 0.0:
        lam = 1.0

    # Base growth: RePlAce-style geometric ramp, but bounded.
    UPPER_PCOF, LOWER_PCOF = 1.05, 1.01
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation:
    #   - High overflow  -> push density weight up faster (spread cells).
    #   - Low overflow   -> ease off so HPWL/wirelength can refine.
    # Use the recent trend in overflow when available.
    trend = 0.0
    if overflow_history:
        recent = [_finite(o, ovf) for o in overflow_history[-5:]]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]   # positive => overflow rising

    if ovf > 0.10:
        # Still congested: accelerate, more so if overflow is climbing.
        mu = base * (1.0 + 0.5 * min(ovf, 1.0) + 2.0 * max(trend, 0.0))
        mu = min(mu, 1.10)
    else:
        # Nearly legal: decay toward gentle growth so we stop over-pushing.
        relief = (0.10 - ovf) / 0.10          # 0 -> 1 as overflow shrinks
        mu = LOWER_PCOF + (base - LOWER_PCOF) * (1.0 - 0.9 * relief)

    # Damp explosive growth when gradients are already large.
    if gnorm > 0.0:
        mu = 1.0 + (mu - 1.0) / (1.0 + 0.001 * gnorm)

    new_lam = lam * mu

    # Hard clamp to the legal range; never return non-finite.
    if new_lam != new_lam or new_lam in (float("inf"), float("-inf")):
        new_lam = lam
    return float(min(max(new_lam, LOWER), UPPER))