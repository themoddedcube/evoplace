def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ ... """
    # --- Sanitize inputs (guard against NaN/inf that produce inf HPWL) ---
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
    cur = min(max(cur, 0.01), 50.0)
    gnorm = _finite(gradient_norm, 1.0)

    # --- Overflow trend: are we still spreading or settling? ---
    hist = overflow_history if isinstance(overflow_history, list) else []
    recent = [_finite(h, ovf) for h in hist[-5:]]
    if len(recent) >= 2:
        trend = recent[-1] - recent[0]          # >0 worsening, <0 improving
    else:
        trend = 0.0

    # --- Base multiplicative growth (DREAMPlace-style ramp on density weight) ---
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.00
    base_mu = UPPER_PCOF * max(0.9999 ** float(it), 0.98)

    # --- Overflow-adaptive modulation -------------------------------------
    # High overflow  -> push lambda up harder to compact the layout.
    # Low overflow   -> ease off so HPWL can be fine-tuned without distortion.
    if ovf > 0.10:
        # Scale growth with how far we are from the overflow target (~0.10).
        push = 1.0 + 0.5 * (ovf - 0.10)         # up to ~1.45x at full overflow
        mu = base_mu * push
        # If overflow is climbing despite the penalty, lean in a bit more.
        if trend > 0.0:
            mu *= 1.05
    else:
        # Near-converged density: anneal the penalty toward stability.
        # Interpolate between gentle growth and slight relaxation.
        frac = ovf / 0.10                        # 0..1 within target band
        mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * frac
        if trend < 0.0:                          # still improving -> relax
            mu *= 0.99

    # --- Gradient-norm safeguard: damp updates when gradients explode -----
    if gnorm > 1e3:
        mu = 1.0 + (mu - 1.0) * 0.5

    # --- Clamp the multiplier to a sane range -----------------------------
    mu = min(max(mu, 0.90), 1.50)

    new_lambda = cur * mu

    # --- Final clamp to the required output range -------------------------
    if new_lambda != new_lambda:                # NaN guard
        new_lambda = cur
    return float(min(max(new_lambda, 0.01), 50.0))