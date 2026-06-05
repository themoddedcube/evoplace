def lambda_schedule(
    iteration: int,            
    overflow: float,           
    overflow_history: list,    
    gradient_norm: float,      
    current_lambda: float,     
) -> float:
    """ Overflow-adaptive multiplicative density-weight schedule. """
    LOWER_PCOF = 0.95
    UPPER_PCOF = 1.05

    # Base DREAMPlace-style decaying growth factor.
    base = UPPER_PCOF * max(0.9999 ** float(iteration), 0.98)

    # Overflow-adaptive modulation: push harder while spreading is poor,
    # ease off as the layout legalizes so HPWL can refine.
    of = overflow if overflow == overflow else 1.0          # guard NaN
    of = min(max(of, 0.0), 1.0)

    # Trend of overflow over the recent history.
    trend = 0.0
    if overflow_history and len(overflow_history) >= 2:
        recent = overflow_history[-5:]
        trend = recent[0] - recent[-1]                      # >0 means improving

    if of > 0.10:
        # Still congested: accelerate, more so if overflow is stalling.
        accel = 1.0 + 0.5 * of
        if trend <= 1e-4:                                   # not improving -> push
            accel *= 1.10
        mu = 1.0 + (base - 1.0) * accel
    else:
        # Nearly legal: throttle growth, gently relax for fine-tuning.
        relax = of / 0.10                                   # 0..1
        mu = LOWER_PCOF + (base - LOWER_PCOF) * relax

    # Gradient-norm safeguard: avoid blowing up when gradients explode.
    if gradient_norm == gradient_norm and gradient_norm > 0.0:
        if gradient_norm > 1e3:
            mu = min(mu, 1.0)

    new_lambda = current_lambda * mu

    # Clamp to the allowed range.
    if new_lambda < 0.01:
        new_lambda = 0.01
    elif new_lambda > 50.0:
        new_lambda = 50.0
    return float(new_lambda)