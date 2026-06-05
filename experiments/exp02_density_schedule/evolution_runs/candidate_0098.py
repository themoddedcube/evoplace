def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive density-weight schedule. """
    UPPER_PCOF = 1.05
    LOWER_PCOF = 1.001

    # Baseline DREAMPlace-style decaying multiplier: aggressive early, gentle late.
    base_mu = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Clamp overflow to a sane range and use it to modulate the multiplier:
    # high overflow (cells still overlapping) -> push lambda up harder,
    # low overflow (nearly legal) -> ease off so HPWL can be fine-tuned.
    of = overflow if overflow == overflow else 1.0      # guard against NaN
    of = max(0.0, min(1.0, of))
    mu = LOWER_PCOF + (base_mu - LOWER_PCOF) * (0.4 + 0.6 * of)

    # Stall detection: if recent overflow is not decreasing, the density force
    # is too weak -> accelerate. If it is dropping fast, decelerate slightly.
    if len(overflow_history) >= 3:
        prev = float(overflow_history[-2])
        prev2 = float(overflow_history[-3])
        recent_delta = prev - of            # >0 means improving
        smoothed = 0.5 * (prev2 - prev) + 0.5 * recent_delta
        if smoothed <= 1e-4 and of > 0.1:
            mu *= 1.03                      # stuck and still illegal -> push
        elif smoothed > 2e-2:
            mu *= 0.99                      # improving quickly -> coast

    # Gradient safety valve: if gradients are blowing up, throttle growth to
    # keep the optimization numerically stable.
    if gradient_norm == gradient_norm and gradient_norm > 1.0e4:
        mu = min(mu, 1.01)

    # Never shrink below ~1 once overflow is significant; allow mild relaxation
    # only in the legal/fine-tuning regime.
    if of > 0.05:
        mu = max(mu, 1.0)
    else:
        mu = max(mu, 0.999)

    new_lambda = current_lambda * mu
    return float(max(0.01, min(50.0, new_lambda)))